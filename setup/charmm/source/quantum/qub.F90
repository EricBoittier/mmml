!     Jiali Gao, 9/15/01
!     Dan T. Major, 01/18/2005 .. 12/2011
!     Questions to: majort@mail.biu.ac.il
module qub_m
  implicit none
contains

#if KEY_QUANTUM==0 && KEY_SCCDFTB==0 && KEY_SQUANTM==0 && KEY_QTURBO==0 && KEY_QCHEM==0 /*qmsetmain*/
SUBROUTINE QUB(COMLYN,COMLEN)
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN

  CALL WRNDIE(-1,'<QUBDEFN>','QUANTUM, SQUANTM, SCCDFTB, GAMESSUK, QTURBO, or QCHEM codes not compiled.')
  RETURN
END SUBROUTINE QUB
#else /* (qmsetmain)*/

SUBROUTINE QUB(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Define the set of quantum mechanical atoms and set up the data
  !     structures.
  !
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use bases_fcm
  use coord
  use deriv
  use psf
  use select
  use stream
  use string
  use sizes
  use number
  use quantm
  use qubpi
#if KEY_SQUANTM==1
  use squantm  
#endif
#if KEY_PARALLEL==1
  use parallel 
#endif
  use gamess_fcm,only:qmused_qchem,qmused_gamess,qmused_turbo
  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  INTEGER   NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD,I,J,BUFSIZ
  integer, allocatable,dimension(:) :: ISLCT
  real(chm_real),allocatable,dimension(:,:):: CBDSX
  real(chm_real),allocatable,dimension(:,:):: CBDSY
  real(chm_real),allocatable,dimension(:,:):: CBDSZ
  real(chm_real),allocatable,dimension(:)  :: RXNCOOR
  real(chm_real),allocatable,dimension(:)  :: RXNPIAV
  real(chm_real),allocatable,dimension(:)  :: RXNPIAV2
  real(chm_real),allocatable,dimension(:)  :: EFACT
  real(chm_real),allocatable,dimension(:)  :: FFACT

#if KEY_PARALLEL==1
#if KEY_MPI==0
  CALL WRNDIE(-3,'<QUBDFN>','Parallel PI needs MPI.')
  QMPI=.FALSE. 
#else /**/
#if KEY_SQUANTM==1
  QMPI = .TRUE.   ! Use specialized parallel QM code only works with SQUANTM
#endif 
#endif 
  if ((numnod>1).and.(qmused_qchem.or.qmused_turbo.or.qmused_gamess)) &
       CALL WRNDIE(-1,'<QUBDFN>', 'Parallel PI is not compatible with GAMESSUK, QCHEM, or QTURBO.')
#endif 

  !
  !  quantum (path integral) particle selection
  !
  call chmalloc('qub.src','QUB','ISLCT',NATOM,intg=ISLCT)
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !
  !------------------------------------------------------------------
  ! Bisection Quantized Classical Path or additional methods check here
  CALL QUBMETHOD(COMLYN,COMLEN)
  ! Read in input options for methods chosen
  CALL QUBDFN(COMLYN,COMLEN,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD)
  !
  CALL TRIMA(COMLYN,COMLEN)
  ! namkh: 12-21-2009: ??? Why mynod.eq.0 ?
#if KEY_PARALLEL==1
  !====      IF(MYNOD.EQ.0) THEN    
#endif
  IF(COMLEN.NE.0) CALL WRNDIE(-1,'<QUBDFN>','Too many characters in PI definition.')
#if KEY_PARALLEL==1
  !====      ENDIF         
#endif
  !
  call chmalloc('qub.src','QUB','EFACT',NBEADSQQ,crl=EFACT)
  call chmalloc('qub.src','QUB','FFACT',NBEADSQQ,crl=FFACT)
  ! Initialize various variables and data structures for path-integrals
  CALL QUBINIT(ISLCT,NATOM,AMASS,WMAIN,EFACT,FFACT)
  ! Initialize beads for path-integrals
  call chmalloc('qub.src','QUB','CBDSX',NBEADSQQ,NPIATM,crl=CBDSX)
  call chmalloc('qub.src','QUB','CBDSY',NBEADSQQ,NPIATM,crl=CBDSY)
  call chmalloc('qub.src','QUB','CBDSZ',NBEADSQQ,NPIATM,crl=CBDSZ)
  ! If parallel, only node 0 deals with beads as random number sequence in Metropolis initial bead generation
  ! won't necessarily be same for different nodes
#if KEY_PARALLEL==1
  if (mynod==0) then
#endif 
  CALL QPINIT(CBDSX,CBDSY,CBDSZ,AMASS)
#if KEY_PARALLEL==1
  endif
  if (numnod.gt.1) then
     BUFSIZ = NBEADSQQ*NPIATM
     CALL PSND8m(CBDSX,BUFSIZ)
     CALL PSND8m(CBDSY,BUFSIZ)
     CALL PSND8m(CBDSZ,BUFSIZ)
  endif
#endif 
  !
  call chmalloc('qub.src','QUB','RXNCOOR',NCRD,crl=RXNCOOR)
  call chmalloc('qub.src','QUB','RXNPIAV',NCRD,crl=RXNPIAV)
  call chmalloc('qub.src','QUB','RXNPIAV2',NCRD,crl=RXNPIAV2)

  ! Switch off gradient for QM calc.
#if KEY_SQUANTM==1
  IF (.NOT. (TIAC.OR.CHAC)) QGRAD = .FALSE.   
#endif
  !
  !  Perform Monte Carlo and Configuration averaging
  !
  CALL QPIRUN(COMLYN,COMLEN,X,Y,Z,DX,DY,DZ,WMAIN,CBDSX,CBDSY,CBDSZ,AMASS,EFACT,FFACT, &
       NATOM,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD,RXNCOOR,RXNPIAV,RXNPIAV2)
  !
  ! deallocate memory
  !
  call chmdealloc('qub.src','QUB','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('qub.src','QUB','CBDSX',NBEADSQQ,NPIATM,crl=CBDSX)
  call chmdealloc('qub.src','QUB','CBDSY',NBEADSQQ,NPIATM,crl=CBDSY)
  call chmdealloc('qub.src','QUB','CBDSZ',NBEADSQQ,NPIATM,crl=CBDSZ)
  call chmdealloc('qub.src','QUB','RXNCOOR',NCRD,crl=RXNCOOR)
  call chmdealloc('qub.src','QUB','RXNPIAV',NCRD,crl=RXNPIAV)
  call chmdealloc('qub.src','QUB','RXNPIAV2',NCRD,crl=RXNPIAV2)
  call chmdealloc('qub.src','QUB','EFACT',NBEADSQQ,crl=EFACT)
  call chmdealloc('qub.src','QUB','FFACT',NBEADSQQ,crl=FFACT)
  ! Switch back on
#if KEY_SQUANTM==1
  QGRAD = .TRUE.  
#endif

#if KEY_PARALLEL==1
  QMPI = .FALSE. 
#endif
  !

  RETURN
END SUBROUTINE QUB

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QUBMETHOD(COMLYN,COMLEN)
  ! Choose path-integral method based on input 
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
  use string
  use number
  use qubpi
  use gamess_fcm
  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  INTEGER   I,J
!------------------------------------------------------------------
! Initially no nuclear QM method chosen
  QNUQM(1:MAXNUQM) = .FALSE.

! Bisection Quantized Classical Path or additional methods check here
  QNUQM(QCP) = (INDXA(COMLYN,COMLEN,'QCP').GT.0)   ! Standard QCP w/Metropolis MC
  QNUQM(BQCP) = (INDXA(COMLYN,COMLEN,'BQCP').GT.0) ! QCP with bisection sampling
  QNUQM(SQCP) = (INDXA(COMLYN,COMLEN,'SQCP').GT.0) ! QCP with staging sampling
  QNUQM(BFEP) = (INDXA(COMLYN,COMLEN,'BFEP').GT.0) ! Mass-perturbation QCP with bisection sampling
  QNUQM(SFEP) = (INDXA(COMLYN,COMLEN,'SFEP').GT.0) ! Mass-perturbation QCP with staging sampling
  QNUQM(QCOP) = (INDXA(COMLYN,COMLEN,'QCOP').GT.0) ! Open-chain path-integral w/staging

  J = 0
  DO I = 1,MAXNUQM
     IF (QNUQM(I)) J = J + 1
  ENDDO
  IF (J.NE.ONE) CALL WRNDIE(0,'<QUBDEFN>','Choose one NUQM method')
  BISE = .FALSE.
  STAGE = .FALSE.
  OSTAGE = .FALSE.
  QFEP = .FALSE.
  if (QNUQM(BQCP) .or. QNUQM(BFEP)) BISE = .TRUE.
  if (QNUQM(SQCP) .or. QNUQM(SFEP)) STAGE = .TRUE.
  if (QNUQM(BFEP) .or. QNUQM(SFEP)) QFEP = .TRUE.
  if (QNUQM(QCOP)) THEN
     BISE = .FALSE.
     STAGE = .TRUE.
     OSTAGE = .TRUE.
     QFEP = .FALSE.   ! Set to F until method worked out
  endif
! WMAIN array conflict
  if (QFEP) then
         IF (QBLUCH) CALL WRNDIE(0,'<QUBDEFN>','PI-FEP/UM and GUK,QCHEM blurred charges are mutually exclusive')
  endif
  if (prnlev.ge.2) then
     IF (QNUQM(QCP)) THEN
         WRITE(OUTU,'(A)') ' QUBDEF> QCP method will be used with the Metropolis algorithm.'
     ELSEIF (QNUQM(BQCP)) THEN
         WRITE(OUTU,'(A)') ' QUBDEF> QCP method will be used with the bisection algorithm.'
     ELSEIF (QNUQM(SQCP)) THEN
         WRITE(OUTU,'(A)') ' QUBDEF> QCP method will be used with the staging algorithm.'
     ELSEIF (QNUQM(BFEP)) THEN
         WRITE(OUTU,'(A)') ' QUBDEF> PI-FEP/UM method will be used with bisection.'
     ELSEIF (QNUQM(SFEP)) THEN
         WRITE(OUTU,'(A)') ' QUBDEF> PI-FEP/UM method will be used with staging.'
     ELSEIF (QNUQM(QCOP)) THEN
         WRITE(OUTU,'(A)') ' QUBDEF> Open chain PI method will be used with staging.'
     ENDIF
  endif
! Check if TI action should be used - works with any sampling method
  TIAC = (INDXA(COMLYN,COMLEN,'TIAC').GT.0)
  if (TIAC) then 
     if (prnlev.ge.2) write(OUTU,'(A)') ' QUBDEF> TI action will be used.'
     IF (QNUQM(QCOP))  CALL WRNDIE(0,'<QUBDEFN>','TI action cannot be used with open chain PI algorithm')
  endif
! Check if Chin action should be used - works with general staging or standard MC only
  CHAC = (INDXA(COMLYN,COMLEN,'CHAC').GT.0)
  IF (CHAC) THEN
     if (prnlev.ge.2) write(OUTU,'(A)')' QUBDEF> Chin action will be used.'
     if (BISE) CALL WRNDIE(0,'<QUBDEFN>','Chin action cannot use bisection algorithm')
     IF (QNUQM(QCOP))  CALL WRNDIE(0,'<QUBDEFN>','Chin action cannot be used with open chain PI algorithm')
  ENDIF
  IF (TIAC.AND.CHAC) CALL WRNDIE(0,'<QUBDEFN>','Choose either PA, TI or Chin action')
  IF (BISE.AND.STAGE) CALL WRNDIE(0,'<QUBDEFN>','Choose either bisection or staging algorithms')
! End methods check

  RETURN
END SUBROUTINE QUBMETHOD

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QUBDFN(COMLYN,COMLEN,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD)
  !-----------------------------------------------------------------------
  !     Read in various user specified values
  ! squantm.a should appear before quantum.a in Makefile on some compilers for
  !
  use ewald,only: lewald
  use chm_kinds
  use dimens_fcm
  use stream
  use string
  use exfunc
  use consta
  use number
  use qubpi
  !
#if KEY_QUANTUM==1
  use sizes
  use quantm
  use qmlinkm
#endif 
#if KEY_SQUANTM==1
  use squantm     
#endif
  !
#if KEY_SCCDFTB==1
  use blockscc_fcm
  use sccdftb
  use sccdftbsrc
  use stbgho
#endif 
  !
#if KEY_SCCDFTB==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1
  use gamess_fcm
#endif 
  !
  use parallel
  use cvio
  use dynio
#if KEY_FLUCQ==1
  use flucq       
#endif
  use machutil,only:daytim
  implicit none
#if KEY_SQUANTM==1
  LOGICAL QGHO_SQ
  INTEGER NQM_GHO,IQLINK(NQMAX)  ! same as maximum qm atoms.
#endif 
  !
  ! Passed variables
  CHARACTER(len=*) COMLYN
  INTEGER          COMLEN
  INTEGER          NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD
  ! Local variables
  INTEGER          DAY,MONTH,YEAR,HOUR,MINUTE,SECOND,ID
  INTEGER          ICNTRL(20),IUNIT,ICRD,ISKIP
  CHARACTER(len=4) HDR
  !
  NFU   = GTRMI(COMLYN, COMLEN, 'FIRS',     -1)
  NUNIT = GTRMI(COMLYN, COMLEN, 'NUNI',      1)
  NBEGN = GTRMI(COMLYN, COMLEN, 'BEGI',      0)
  NSTOP = GTRMI(COMLYN, COMLEN, 'STOP',      0)
  NSKIP = GTRMI(COMLYN, COMLEN, 'SKIP',      1)
  !
  FPITEMP    = GTRMF(COMLYN,COMLEN,'TEMP',298.15_chm_real)
  MCEQUIL    = GTRMI(COMLYN,COMLEN,'MCEQ',10)
  MCCONF     = GTRMI(COMLYN,COMLEN,'MCON',10)
  NBEADSQ    = GTRMI(COMLYN,COMLEN,'BEAD',32)
  NBMOVE     = GTRMI(COMLYN,COMLEN,'NBMO',1)
  NBIN_UNIT  = GTRMI(COMLYN,COMLEN,'BDIN',0)
  NBOUT_UNIT = GTRMI(COMLYN,COMLEN,'BDOU',0)
  NBPDB_UNIT = GTRMI(COMLYN,COMLEN,'BPDB',0)
  NQUBOUT    = GTRMI(COMLYN,COMLEN,'OUNI',6)
  NQUBOUT2   = GTRMI(COMLYN,COMLEN,'OUNJ',6)
  NMOMDIS    = GTRMI(COMLYN,COMLEN,'OUNK',6)
  NMOMDIS2   = GTRMI(COMLYN,COMLEN,'OUNL',6)
  IRN        = GTRMI(COMLYN,COMLEN,'IRAN',-1)
  !  BISE       = (INDXA(COMLYN,COMLEN,'BISE').GT.0)
  KLEV       = GTRMI(COMLYN,COMLEN,'KLEV',5)
  FENER      = (INDXA(COMLYN,COMLEN,'FAST').GT.0)
  FFOCK      = (INDXA(COMLYN,COMLEN,'FFOC').GT.0)
  NOEWALD    = (INDXA(COMLYN,COMLEN,'NOEW').GT.0)
  ! Enforce no Ewald due to specialized parallel routines
#if KEY_SQUANTM==1
  if (numnod > 1) NOEWALD = .true.
#endif 
  ! In case of open PI increase effective bead number by one as it samples P+1 beads
  ! In all equations, the number of beads is still P
  IF (OSTAGE) THEN
     NBEADSQQ = NBEADSQ + 1   ! Open chain PI used P+1 beads
  ELSE
     NBEADSQQ = NBEADSQ
  ENDIF
  QRXN = (INDXA(COMLYN,COMLEN,'QRXN').GT.0)
  ISTR = .FALSE.              ! Isotropic sampling for open chain PI initialized to false
  !
  !  Exclude use of other incompatible
#if KEY_QUANTUM==1
  IF (QMPERT) CALL WRNDIE(-1,'<QUBDFN>','Path-integral is not compatible with PERT.')
  IF (QDECOM) CALL WRNDIE(-1,'<QUBDFN>','Path-integral is not compatible with DECO.')
  IF (CHDYN)  CALL WRNDIE(-1,'<QUBDFN>','Path-integral is not compatible with CHDY.')
#endif 
#if KEY_SCCDFTB==1
  IF (QLAMDA) CALL WRNDIE(-1,'<QUBDFN>','Path-integral is not compatible with QLAMDA.')
  IF (QPKAC)  CALL WRNDIE(-1,'<QUBDFN>','Path-integral is not compatible with QPKAC.')
  IF (LSCCRP) CALL WRNDIE(-1,'<QUBDFN>','Path-integral is not compatible with NSCCRP.')
  !      IF (FENER) LGRAD = .FALSE.             ! Switch off gradient calc
#endif 
#if KEY_SQUANTM==1
  QGMREM = .TRUE.
  If(Prnlev.ge.2) WRITE(OUTU,'(A)') ' QUBDEF> REMOve: Classical energies within QM atoms are removed.'
  ! for updating GHO atom information.
  QGHO_SQ=.false.
  NQM_GHO= 0
  Call Get_gho_info(NQMAX,NQM_GHO,IQLINK,QGHO_SQ)
#endif 
  !
#if KEY_FLUCQ==1
  IF (QFLUC) CALL WRNDIE(-2,'<QUBDFN>','Path-integral is not compatible with FLUCQ.')
#endif 

  !  If Fast Fock matrix updating requested, switch on fast energy evaluation - only for old mopac code
  IF (FFOCK) FENER = .TRUE.
  !  Disable Ewald - long-range effects not important for NQM
  IF (NOEWALD) THEN
     LEWALD = .FALSE.
#if KEY_QUANTUM==1 || KEY_SQUANTM==1
     LQMEWD = .FALSE.
#endif 
#if KEY_SCCDFTB==1
     ! ask QC what if periodic without Ewald
     PERIOD = .FALSE.
     QSCCNB = .TRUE.
#endif 
  ENDIF
  ! Enforce defaults for bisection methods - otherwise incorrect
  IF (BISE) THEN
     IF ((2**KLEV).NE.NBEADSQ) THEN
        CALL WRNDIE(0,'<QUBDFN>','2**KLEV must equal NBEADSQ')
     ENDIF
  ENDIF
  !  Check if reaction coordinate info provided
  IF (QRXN) THEN
     RXNA = GTRMI(COMLYN,COMLEN,'RXNA', 99999)
     RXNB = GTRMI(COMLYN,COMLEN,'RXNB', 99999)
     RXNC = GTRMI(COMLYN,COMLEN,'RXNC', 99999)
     IF((RXNA.EQ.99999).OR.(RXNB.EQ.99999).OR.(RXNC.EQ.99999))  &
          CALL WRNDIE(0,'<QUBDFN>','choose 3 atoms')
     BINWIDTH = GTRMF(COMLYN,COMLEN,'DELD', 0.01D0)
     RESLN = GTRMF(COMLYN,COMLEN,'RESL', 10.0D0)
     IF (OSTAGE) ISTR = (INDXA(COMLYN,COMLEN,'ISTR').GT.0)   ! Isotropic sampling along A---C axis
     IF (ISTR .and. prnlev.ge.2) write(OUTU,'(A)')' QUBDEF> Isotropic sampling will be used.'
  ENDIF
  ! Various warnings
  IF (NQUBOUT.LE.0) CALL WRNDIE(0,'<QUBDFN>','choose an output unit')
  IF (QFEP.AND.NQUBOUT2.LE.0) CALL WRNDIE(0,'<QUBDFN>','choose an output unit')
  IF (OSTAGE.AND.NMOMDIS.LE.0) CALL WRNDIE(0,'<QUBDFN>','choose an output unit')
  IF (OSTAGE.AND.QFEP.AND.NMOMDIS2.LE.0) CALL WRNDIE(0,'<QUBDFN>','choose an output unit')
  IF (MCCONF.LE.0) CALL WRNDIE(0,'<QUBDFN>','MC move not defined')
  IF (NBMOVE.LT.1 .OR. NBMOVE.GT.NBEADSQ) CALL WRNDIE(-1,'<QUBDFN>','2 <= NBMO <= NBEADSQ')
  ! Lower default for QCP changed to 2 - with 1 & centroid constraint meaningless
  IF (NBMOVE.EQ.1 .AND. QNUQM(QCP)) CALL WRNDIE(-1,'<QUBDFN>','Metropolis sampling requires NBMO > 1')
  IF (NBEADSQ.LT.MINBEADS .OR. NBEADSQ.GT.MAXBEADS) CALL WRNDIE(-1,'<QUBDFN>', &
       'MINBEADS <= number of PI beads <= MAXBEADS')
#if KEY_PARALLEL==1
  IF (BISE.or.STAGE) THEN
     IF (NBEADSQQ.LT.NUMNOD) CALL WRNDIE(-1,'<QUBDFN>','NUMNOD <= NBEADSQ')
  ELSE
     IF (NUMNOD.GT.1 .and. NBMOVE.NE.NUMNOD) CALL WRNDIE(-1,'<QUBDFN>','NUMNOD = NBMO')
  ENDIF
#endif 
  IF (BISE) THEN
     IF(KLEV.LT.1 .OR. 2**KLEV.GT.NBEADSQ) CALL WRNDIE(-1,'<QUBDFN>','2 <= 2**KLEV <= NBEADSQ')
  ENDIF
  ! Number of beads with Chin must be multiple of 3 due to factorization
  IF (CHAC) THEN
     IF (MOD(DBLE(NBEADSQ),3.0D0).NE.ZERO) CALL WRNDIE(-1,'<QUBDFN>', &
        'Number of beads with Chin method must be multiple of 3')
  ENDIF
  !   Generate time-dependent random # generator seed (must be odd)
  ID = 0
  IF (IRN .LE. 0) THEN
     ! Use system independent function
     CALL DAYTIM(MONTH,DAY,YEAR,HOUR,MINUTE,SECOND)
     ID = HOUR*3600 + MINUTE*60 + SECOND + 314159165
     IRN = MOD(ID,1000000)
     IF (MOD(IRN,2) .EQ. 0) IRN = IRN + 1
  ENDIF
  !  debug with const irn
  !     IRN = 99999
  if(prnlev.ge.2) WRITE(OUTU,'(2(A,I12))') ' QUBDEF> ID = ',ID,'  IRN = ',IRN
  IF (BISE .or. STAGE) THEN
     IF (NBMOVE .GT. 1) CALL WRNDIE(-1,'<QUBDFN>','NBMO=1 when using bisection or staging')
     NBMOVE = 1  ! Enforce
  !     if(prnlev.ge.2) WRITE(OUTU,'(A)') ' QUBDEF> MC fp sampling will use Bisection Algorithm.'
  ELSE
     PIMCMV = GTRMF(COMLYN,COMLEN,'MCMV',1.0D0)
     if (prnlev.ge.2) then
  !      WRITE(OUTU,'(A)') ' QUBDEF> MC fp sampling will use Metropolis.'
        WRITE(OUTU,'(A,F10.5)') ' QUBDEF> MCMV = ',PIMCMV
     end if
  ENDIF
  !
  ! Get dimensions of trajectory file(s)
  ! ICNTRL(4)=number of time steps in traj file, ICNTRL(3)=freq of save
  ! Number of crd frames in traj file is ICNTRL(1)=ICNTRL(4)/ICNTRL(3)
  !
  ICRD = ZERO
  ISKIP = NSKIP
  DO IUNIT = NFU,NFU+NUNIT-1
     CALL GTICNT(IUNIT,HDR,ICNTRL,.TRUE.,.TRUE.)
     IF (MOD(ICNTRL(4),ISKIP).NE.ZERO .OR. ISKIP.LT.ICNTRL(3))  ISKIP = ICNTRL(3)
     ICRD = ICRD + ICNTRL(4)/ISKIP
     !     if(prnlev.ge.2) write(*,*) 'icntrl',ICNTRL
     !     if(prnlev.ge.2) write(*,*) 'ncrd',ICRD,ISKIP
  ENDDO
  NCRD = ICRD
  !
  ! Write out some information and options requested.
  !
  if (prnlev.ge.2) then
     IF (FENER) WRITE(OUTU,'(A)') ' QUBDEF> Fast QM routine will be used.'
     IF (FFOCK) WRITE(OUTU,'(A)') ' QUBDEF> Fast Fock matrix updating routine will be used.'
     IF (NOEWALD) WRITE(OUTU,'(A)')' QUBDEF> Ewald summation will NOT be used.'
     WRITE(OUTU,'(A)') ' QUBDEF> Some nuclei will be treated quantum mechanically.'
     WRITE(OUTU,'(A,I7)') ' QUBDEF> The number of quasi-particles per atom  = ',NBEADSQ
     WRITE(OUTU,'(A,I7)') ' QUBDEF> The number of Monte Carlo moves (av)    = ',MCCONF
     WRITE(OUTU,'(A,I7)') ' QUBDEF> The number of Monte Carlo moves (eq)    = ',MCEQUIL 
  endif
  !
  !     CALL TRIMA(COMLYN,COMLEN)
  !
  RETURN
END SUBROUTINE QUBDFN

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QUBINIT(ISLCT,NATOM,AMASS,WMAIN,EFACT,FFACT)
  !-----------------------------------------------------------------------
  !     The set of atoms whose energies will be calculated quantum
  !     mechanically are defined here and the appropriate data structures
  !     set up.
  !
  ! squantm.a should appear before quantum.a in Makefile on some compilers for
  ! QMMM_MODULE to be recognized
  !...##IF SQUANTM
  !      USE QMMM_MODULE, ONLY : QM2_GHOS
  !...##ENDIF
  !
  use ewald,only: lewald
  use chm_kinds
  use dimens_fcm
  use stream
  use string
  use exfunc
  use consta
  use number
  use qubpi
  !
#if KEY_QUANTUM==1
  use sizes
  use quantm
  use qmlinkm
#endif 
#if KEY_SQUANTM==1
  use squantm     
#endif
  !
#if KEY_SCCDFTB==1
 use blockscc_fcm
 use sccdftb
  use sccdftbsrc
  use stbgho
#endif 
  !
  use gamess_fcm
  !
  use parallel
  use cvio
  use dynio
#if KEY_FLUCQ==1
  use flucq       
#endif
  use machutil,only:daytim
  implicit none
  ! Passed variables
  INTEGER          ISLCT(*),NATOM
  real(chm_real)   AMASS(*),WMAIN(*),EFACT(NBEADSQQ),FFACT(NBEADSQQ)
  ! Local variables
  real(chm_real)   LAMBDA_TMP
  INTEGER          I,J,K,L
#if KEY_SQUANTM==1
  LOGICAL QGHO_SQ
  INTEGER NQM_GHO,IQLINK(NQMAX)  ! same as maximum qm atoms.
#endif 

  BETA_FPI   = KCALMOL/(FPITEMP*JKBOLTZ)
  LAMBDA_TMP = ANGSTROM*JKBOLTZ*FPITEMP/HBAR
  FPIFAC     = half*DBLE(NBEADSQ)*AMU*LAMBDA_TMP*LAMBDA_TMP/KCALMOL
  LAMBDA_TMP = HBAR/(two*LAMBDA_TMP*ANGSTROM*DBLE(NBEADSQ)*AMU)
  ! Initialize Chin and TI data structures
  IF (TIAC .OR. CHAC) THEN
     DO I = 1,NBEADSQQ
        EFACT(I) = ONE
        FFACT(I) = ONE
     ENDDO
  ENDIF
  CHFACN = ONE   ! Should be one if not using Chin
  IF (CHAC) CALL QPIPARAM(LAMBDA_TMP,EFACT,FFACT)

  !     call gflush(6)
  !
  !=================
  ! PI atoms may not be GHO atoms. PI atoms should not have any
  ! MM bonded terms (e.g. within 2 bonds of GHO atoms). These terms
  ! are not accounted for (if using QUANTUM or SQUANTM). 
#if KEY_QUANTUM==1
  IF (MQMLNK.GT. ZERO) WRITE(OUTU,'(A)')' QUBDEF> No MM bonded terms are included.'  
#endif
#if KEY_SQUANTM==1
  IF (NQM_GHO.GT.ZERO) WRITE(OUTU,'(A)')' QUBDEF> No MM bonded terms are included.'  
#endif
#if KEY_SCCDFTB==1
  IF (NQMLNK.GT. ZERO) WRITE(OUTU,'(A)')' QUBDEF> No MM bonded terms are included.'  
#endif
  !
  !  Determine the number of nuclear quantum mechanical (PI) atoms
  !
  NPIATM= 0
  J = 0
  K = 1
  DO I = 1,NATOM
#if KEY_QUANTUM==1
     IF (QATLAB(I).GT.0) J = J + 1     
#endif
#if KEY_SCCDFTB==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1
     IF (IGMSEL(I).GT.0) J = J + 1     
#endif
     IF(ISLCT(I) .EQ. 1) THEN
        NPIATM= NPIATM + 1
        IPIATM(NPIATM) = I
#if KEY_QUANTUM==1
        ! Create auxiliary array of QM PI atoms. Needed for Fast Fock matrix updating.
        IF (QATLAB(I).GT.0) THEN
           DO L=1,MQMLNK
              IF (IMQLINK(L).EQ.I) CALL WRNDIE(0,'<QUBDFN>','PI atom cannot be a link atom.')
           ENDDO
           IQMPIATM(J) = 0
           IF (IPIATM(K) .EQ. I) THEN
              IQMPIATM(J) = 1
              K = K + 1
           ENDIF
           !              if(prnlev.ge.2) write(*,*)I,QATLAB(I),J,IPIATM(J),IQMPIATM(J)
        ELSE
           CALL WRNDIE(0,'<QUBDFN>','PI atom is not a QM atom.')
        ENDIF
#endif 
#if KEY_SCCDFTB==1 || KEY_SQUANTM==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1
        ! IGMSEL = 1, qm atom, gho atom..etc
        ! IGMSEL = 2, h-link atom,
        ! IGMSEL = 5, mm atom excluded from qm/mm non-bonded interactions.
        IF (IGMSEL(I).GT.0) THEN
#if KEY_SCCDFTB==1
           IF (IGHOSL(I).EQ.1 .OR. IGMSEL(I).EQ.2) CALL WRNDIE(0,'<QUBDFN>','PI atom cannot be a link/gho atom.')
#endif 
           IF ((IGMSEL(I).EQ.2).and.(qmused_qchem.or.qmused_turbo)) &
                CALL WRNDIE(0,'<QUBDFN>','PI atom cannot be a link/gho atom.')
#if KEY_SQUANTM==1
           IF (IGMSEL(I).EQ.1) THEN
              IF (QGHO_SQ) THEN
                 DO J=1,NQM_GHO
                    IF (I.EQ. IQLINK(J)) CALL WRNDIE(0,'<QUBDFN>','PI atom cannot be a link/gho atom.')   ! it is gho atom
                 ENDDO
              ENDIF
           ELSEIF (IGMSEL(I).EQ.2) THEN
              CALL WRNDIE(0,'<QUBDFN>','PI atom cannot be a link/gho atom.')   ! it is h-link atom.
           ENDIF
#endif 
        ELSE
           CALL  WRNDIE(0,'<QUBDFN>','PI atom is not a QM atom.')
        ENDIF
#endif 
        ! Compute De Broglie wl accounting for multiple bead moves
        LAMBDA0(NPIATM) = LAMBDA_TMP/AMASS(I)
        ! Mass perturbation - Compute square root mass ratio
        ! User MUST set WMAIN vector in input script
        IF (QFEP) THEN
           IF (WMAIN(I).LE.0) THEN
              CALL  WRNDIE(0,'<QUBDFN>','Set WMAIN for QFEP calc')
           ELSE
              MSQRAT(NPIATM) = SQRT(AMASS(I)/WMAIN(I))
           ENDIF
        ELSE
           MSQRAT(NPIATM) = one
        ENDIF
        !           IF(PRNLEV.GT.4) WRITE(OUTU,'(A,I4,I8,2F10.5,I4)') ' QUBDEF> ',NPIATM,I,AMASS(I),LAMBDA0(NPIATM),IAC(I)
     ENDIF         ! ISLCT
  ENDDO            ! NATOM
  !
  if (prnlev.ge.2) then
     WRITE(OUTU,'(A,I7)') ' QUBDEF> The number of QM path integral atoms    = ',NPIATM
     WRITE(OUTU,'(A,F10.5)') ' QUBDEF> BETA_FPI = ',BETA_FPI
     WRITE(OUTU,'(A,F10.5)') ' QUBDEF> FPIFAC = ',FPIFAC
  end if
  IF (NPIATM.LE.0) CALL WRNDIE(0,'<QUBDFN>','No path integral QM atoms selected.')
  IF (NPIATM.GT.MAXPIATM) CALL WRNDIE(0,'<QUBDFN>','Too many path integral QM atoms selected.')
  IF (NPIATM.GT.1 .AND. OSTAGE) CALL WRNDIE(0,'<QUBDFN>','Only one path integral QM atom with QCOP.')
  !  Compute Thermal De Broglie wavelength
  DO I = 1,NPIATM 
     ! Compute De Broglie wl accounting for multiple pi atoms bead moves  
     !         LAMBDA(I) = SQRT(LAMBDA0(I)/(DBLE(NBMOVE-1)*DBLE(NPIATM))) 
     LAMBDA(I) = SQRT(LAMBDA0(I)/(DBLE(NBMOVE))) 
     LAMBDA2(I) = two*LAMBDA(I) 
     IF (TIAC) HOFAC(I) = (BETA_FPI*HBAR)**2/ &
                          (24.0D0*AMASS(IPIATM(I))*AMU*KCALMOL*(ANGSTROM*DBLE(NBEADSQ))**2)
  !       IF (TIAC) WRITE(6,'(A,I8,E15.5)')'tifac=',i,hofac(i)
     IF (CHAC) THEN
        HOFAC(I) = U0*(BETA_FPI*HBAR)**2/ &
                   (AMASS(IPIATM(I))*AMU*KCALMOL*(ANGSTROM*DBLE(NBEADSQ)*CHFACN)**2)
  ! Modify DeBroglie wavelength for Chin action
        LAMBDA(I) = LAMBDA(I)*SQRT(T1)
        LAMBDA2(I) = two*LAMBDA(I)
     ENDIF

     if(prnlev.ge.2) WRITE(OUTU,'(A,I8,F12.5,F12.5)') ' QUBDEF> LAMBDA,MSQRAT   = ',I,LAMBDA(I),MSQRAT(I) 
  ENDDO

  !
  RETURN
END SUBROUTINE QUBINIT
!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QPIPARAM(LAMBDA_TMP,EFACT,FFACT)
! Compute parameters needed for Chin based methods
  use chm_kinds
  use number
  use qubpi
  implicit none
!
  real(chm_real) LAMBDA_TMP
  real(chm_real) EFACT(NBEADSQQ),FFACT(NBEADSQQ)
!
  INTEGER I
  real(chm_real) V1,V2,A1

! Chin parameters
! Use symmetric T0 parameter - this allows using regular staging algorithm to sample beads
  T0 = sixth
  T1 = half-T0
  V1 = one/(six*((one-two*T0)**2))
  V2 = one-two*V1
  !      A1=T0
  A1 = (one+six*T0*(-three+four*T0*(six+T0*(-23.0D0+24.0D0*T0)))) / &
       (ten*(one-twelve*T0*(one-two*T0)**2)*(one-six*T0*(one+two*T0-four*T0**2)))
  ! U0 is exact up to 6th order for harmonic oscillator.
  ! Scuro and Chin. Phys. Rev. E 2005, 71, 056703.
  U0 = (one/twelve)*(one-(one/(one-two*T0))+(one/(six*((one-two*T0)**3))))
  CHFACN = T1                ! Factorization level is 3 for Chin action
  FPIFAC = FPIFAC*CHFACN     ! Correct for effective # of beads
  LAMBDA_TMP = LAMBDA_TMP/CHFACN  ! Correct for effective # of beads
! Build arrays with Chin parameters for quick lookup
  DO I = 1,NBEADSQ-2,3
     EFACT(I) = V1
     EFACT(I+1) = V2
     EFACT(I+2) = V1
     FFACT(I) = A1
     FFACT(I+1) = one-two*A1
     FFACT(I+2) = A1
  ENDDO
  IF (OSTAGE) THEN   ! In case of open chain path-integral bead P+1 must be set
     EFACT(NBEADSQQ) = EFACT(1)
     FFACT(NBEADSQQ) = FFACT(1)
  ENDIF
  RETURN
  END SUBROUTINE QPIPARAM

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QPINIT(CBDSX,CBDSY,CBDSZ,AMASS)
  use chm_kinds
  use stream
  use sizes
  use qubpi
  implicit none
  !
  real(chm_real) CBDSX(NBEADSQQ,NPIATM),CBDSY(NBEADSQQ,NPIATM),CBDSZ(NBEADSQQ,NPIATM),AMASS(*)
  !
  INTEGER  I,J
  !
  !  generate initial guess of beads positions (if not provided)
  !
  IF(NBIN_UNIT.GT.0) THEN
     READ(NBIN_UNIT,*) I,J
     IF(I.NE.NPIATM .OR. J.NE.NBEADSQQ) CALL WRNDIE(0,'<QUBDFN>', &
          'number of PI atoms and beads not consistent')
     DO I = 1,NPIATM
        READ(NBIN_UNIT,'(6F12.8)') (CBDSX(J,I),CBDSY(J,I), &
             CBDSZ(J,I),J=1,NBEADSQQ)
     ENDDO
  ELSE
     IF (QNUQM(QCP)) THEN
        CALL PIBGEN2(CBDSX,CBDSY,CBDSZ,AMASS,NPIATM,NBEADSQ,NBEADSQQ,IPIATM)
     ELSE
        CALL PIBGEN(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ)
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE QPINIT
!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QPIRUN(COMLYN,COMLEN,X,Y,Z,DX,DY,DZ,WMAIN,CBDSX,CBDSY,CBDSZ,AMASS,EFACT,FFACT,NATOM,  &
                  NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD,RXNCOOR,RXNPIAV,RXNPIAV2)
  use chm_kinds
  use dimens_fcm
  use exfunc
  use memory
  use stream
  use number
  use cvio
  use ctitla
  use sizes
  use qubpi
  use psf, only : CG
  !
#if KEY_PARALLEL==1
  use parallel  
#endif
  implicit none

  ! Passed variables
  INTEGER  COMLEN
  CHARACTER(len=*) COMLYN
  INTEGER :: NATOM,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NCRD                                              
  real(chm_real) :: X(:), Y(:), Z(:), DX(:), DY(:), DZ(:), WMAIN(*), AMASS(*)
  real(chm_real)   CBDSX(NBEADSQQ,NPIATM),CBDSY(NBEADSQQ,NPIATM),CBDSZ(NBEADSQQ,NPIATM)
  real(chm_real)   EFACT(NBEADSQQ),FFACT(NBEADSQQ),RXNCOOR(NCRD),RXNPIAV(NCRD),RXNPIAV2(NCRD)
  ! Local variables                                                                                  
  INTEGER :: NAT,IUNIT
  INTEGER :: NFILE,ISTEP,ISTATS,NDEGF,NSAVV,NFREAT
  real(chm_real)  DELTA,CLNUM
  real(chm_real)  PIAVE,FPITOT,FPITOTSQ,FPIAVE,FPISTD
  real(chm_real)  PIAVE2,FPITOT2,FPITOTSQ2,FPIAVE2,FPISTD2
  CHARACTER(LEN=4),save :: HDRC='CORD', HDRD='VELD'

  real(chm_real4),allocatable,dimension(:) :: ITEMP
  integer,allocatable,dimension(:) :: IFREAT
  integer,allocatable,dimension(:) :: IBMOVE

  logical :: q_pll   ! If parallel run set to true
  logical :: lerr
  !
  NAT = NATOM
  call chmalloc('qub.src','QPIRUN','ITEMP',NATOM,cr4=ITEMP)
  call chmalloc('qub.src','QPIRUN','IFREAT',NATOM,intg=IFREAT)
  call chmalloc('qub.src','QPIRUN','IBMOVE',NBEADSQQ,intg=IBMOVE)

  call MCATMS(IBMOVE,NBEADSQQ)   ! Initialize beads move array

  NFREAT = 0
  IUNIT = NFU
  CLNUM = ZERO
  PIAVE = ZERO
  PIAVE2 = ZERO
  FPITOT = ZERO
  FPITOT2 = ZERO
  FPITOTSQ = ZERO
  FPITOTSQ2 = ZERO
  q_pll = .true.   ! Read as if serial even if parallel
  !
  ! Iterate over different classical configurations
  istats = 1
  loopwhile: Do while (istats >=0)
     CLNUM = CLNUM+ONE                                                                             
     !
     !  Read in CL coordinates
     !
     CALL READCV(X,Y,Z,    &
!#if KEY_CHEQ==1
!          (/ ZERO /), .FALSE., &  
!#endif
#if KEY_CHEQ==1
          CG,.FALSE., &  
#endif
          ITEMP,NAT,IFREAT,NFREAT,NFU,  &
          NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF,  &
          DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD,  &
          TITLEB,NTITLB,.FALSE., (/ ZERO /), q_pll)
     !
     !  Generate density matrix from scratch for each new structure
     !
#if KEY_QUANTUM==1
     CALL GESDEN(.FALSE.)  
#endif
     !
     !  Call to update qm/mm list
     !  mod MG 9/18/01
     CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
          .TRUE.,.TRUE.,.TRUE.,.TRUE., &
          0,[zero],[zero],[zero],[zero],[zero],[zero])

     if(prnlev.ge.2) WRITE(OUTU,'(/,A,I12)') ' QPIRUN> Classical conf # ',INT(CLNUM)
     !
     !  Perform Monte Carlo Path Integral for configuration
     !
     IF (QNUQM(QCP).OR.QNUQM(BQCP).OR.QNUQM(SQCP).OR.QNUQM(BFEP).OR.QNUQM(SFEP).OR.QNUQM(QCOP)) THEN
#if KEY_PARALLEL==1
        CALL QMPIMC(X,Y,Z,DX,DY,DZ,CBDSX,CBDSY,CBDSZ,AMASS,WMAIN,EFACT,FFACT,NATOM,IBMOVE,PIAVE,PIAVE2)
#else /**/
        CALL QPIMC(X,Y,Z,DX,DY,DZ,CBDSX,CBDSY,CBDSZ,AMASS,WMAIN,EFACT,FFACT,NATOM,IBMOVE,PIAVE,PIAVE2)
#endif 
        !         ELSEIF (QNUQM(...)) THEN
        ! Add call to different NQM PI methods
        !            CALL ...(...)
     ENDIF
     !
     ! namkh: 12-21-2009: moved downward to avoid STOP...     
     !#if KEY_PARALLEL==1
     !         IF(MYNOD.EQ.0) THEN
     !#endif
     IF(PIAVE.LE.ZERO) THEN
        CALL WRNDIE(-1,'<QPIRUN>','PI average is wrong')
        CLNUM=CLNUM-ONE
        STOP
     ENDIF

#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif 
        FPITOT = FPITOT+PIAVE
        FPITOTSQ = FPITOTSQ+PIAVE**2
        FPIAVE = -LOG(PIAVE)/BETA_FPI
        IF (QFEP) THEN
           FPITOT2  = FPITOT2+PIAVE2
           FPITOTSQ2= FPITOTSQ2+PIAVE2**2
           FPIAVE2  =-LOG(PIAVE2)/BETA_FPI
        ENDIF
        !  Extract reaction coordinate related data
        IF (QRXN) THEN
           RXNCOOR(INT(CLNUM)) = GETRXN(X,Y,Z,RXNA,RXNB,RXNC)
           RXNPIAV(INT(CLNUM)) = PIAVE
           WRITE(NQUBOUT,96) RXNCOOR(INT(CLNUM)),PIAVE
           WRITE(OUTU,97) RXNCOOR(INT(CLNUM)),FPIAVE
           IF (QFEP) THEN
              RXNPIAV2(INT(CLNUM)) = PIAVE2
              WRITE(NQUBOUT2,96) RXNCOOR(INT(CLNUM)),PIAVE2
              WRITE(OUTU,97) RXNCOOR(INT(CLNUM)),FPIAVE2
           ENDIF
        ELSE
           WRITE(OUTU,98) FPIAVE
           IF (QFEP) WRITE(OUTU,98) FPIAVE2
        ENDIF
        !
        !  write out all beads coordinates to pdb file
        !
        IF(NBPDB_UNIT.GT.0 .AND. (BISE.or.STAGE)) CALL BWRITE(NBPDB_UNIT,X,Y,Z,CBDSX,CBDSY,CBDSZ, &
             NPIATM,NBEADSQQ,IPIATM,.TRUE.,INT(CLNUM))

#if KEY_PARALLEL==1
     ENDIF
#endif 
  End do loopwhile     ! outer while

! Compute averages and standard deviations
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif 
     IF (CLNUM.LE.one) THEN
        FPISTD = zero
        IF (QFEP) FPISTD2 = zero
     ELSE
        FPISTD = SQRT((FPITOTSQ-FPITOT**2/CLNUM)/CLNUM)
        IF (QFEP) FPISTD2 = SQRT((FPITOTSQ2-FPITOT2**2/CLNUM)/CLNUM)
        !            FPISTD = -LOG(FPISTD)/BETA_FPI
     ENDIF

     FPITOT = FPITOT/CLNUM
     FPIAVE = -LOG(FPITOT)/BETA_FPI                                                 
     ! Standard deviation of ln(x) is dx/x
     FPISTD = FPISTD/(FPITOT*BETA_FPI)
     !         WRITE(NQUBOUT,99) FPIAVE,FPISTD,INT(CLNUM) 
     WRITE(OUTU,99) FPIAVE,FPISTD,INT(CLNUM)
     IF (QFEP) THEN
        FPITOT2 = FPITOT2/CLNUM
        FPIAVE2 = -LOG(FPITOT2)/BETA_FPI
        ! Standard deviation of ln(x) is dx/x
        FPISTD2 = FPISTD2/(FPITOT2*BETA_FPI)
        WRITE(OUTU,99) FPIAVE2,FPISTD2,INT(CLNUM)
     ENDIF

     IF (QRXN) THEN
        CALL QUBIN(RXNCOOR,RXNPIAV,INT(CLNUM))
        IF (QFEP) CALL QUBIN(RXNCOOR,RXNPIAV2,INT(CLNUM))
     ENDIF
     !
     !  Finalize writing to pdb file
     !
     IF(NBPDB_UNIT.GT.0 .AND. (QNUQM(QCP).AND.QNUQM(BQCP))) THEN
        CALL BWRITE(NBPDB_UNIT,X,Y,Z,CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IPIATM,.FALSE.,INT(CLNUM))
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 

  ! deallocate
  call chmdealloc('qub.src','QPIRUN','ITEMP',NATOM,cr4=ITEMP)
  call chmdealloc('qub.src','QPIRUN','IFREAT',NATOM,intg=IFREAT)
  call chmdealloc('qub.src','QPIRUN','IBMOVE',NBEADSQQ,intg=IBMOVE)

96 FORMAT(1X,F8.4,5X,E15.8)
97 FORMAT(1X,'QPIRUN> R =',1X,F8.4,5X,'Del G(CL->QM)  = ',F12.5)
98 FORMAT(1X,'QPIRUN> Del G(CL->QM)  = ',F12.5)
99 FORMAT(1X,'QPIRUN> Del G(CL->QM)tot  = ',F12.5,1X,'+/-',1X, &
       F12.5,/,9X,'for a total of',1X,I7,1X,'classical (centroid) configurations',//)

  RETURN
END SUBROUTINE QPIRUN

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QPIMC(X,Y,Z,DX,DY,DZ,CBDSX,CBDSY,CBDSZ,AMASS,WMAIN,EFACT,FFACT,NATOM,IBMOVE,PIAVE,PIAVE2)
  !
  ! Perform closed or open chain polymer path-integral simulation using Monte Carlo free particle sampling
  !
  use chm_kinds
  use memory
  use stream
  use number
  use dimens_fcm
  use sizes
  use qubpi
  implicit none

  real(chm_real) :: X(:), Y(:), Z(:), DX(:), DY(:), DZ(:)
  real(chm_real)   CBDSX(NBEADSQQ,NPIATM),CBDSY(NBEADSQQ,NPIATM),CBDSZ(NBEADSQQ,NPIATM),AMASS(*),WMAIN(*)
  real(chm_real)   EFACT(NBEADSQQ),FFACT(NBEADSQQ)
  real(chm_real)   PIAVE,PIAVE2
  INTEGER ::       IBMOVE(NBEADSQQ),NATOM

  INTEGER I,J
  INTEGER JJ,KK,ITYP,NCONFS,NCON,NBEADMV,NBMV
  real(chm_real)  PISUM,XACCEPT,XREJECT,XREPEAT
  real(chm_real)  ETOL_FPI,ETNE_FPI
  real(chm_real)  RR,ECL_XBAR,EFPIHOLD
  real(chm_real)  EFPIHNEW,DELH,BFACT,XTMP
  real(chm_real)  XOLD(NPIATM),YOLD(NPIATM),ZOLD(NPIATM),EFPIOLD(NBEADSQQ),EFPIMOV(NBEADSQQ)
  real(chm_real)  XBOLD(NBEADSQQ,NPIATM),YBOLD(NBEADSQQ,NPIATM),ZBOLD(NBEADSQQ,NPIATM)
  real(chm_real)  EHARMAVE,EPOTAVE
  ! Mass-perturbation variables
  real(chm_real)  PCBDSX(NBEADSQQ,NPIATM),PCBDSY(NBEADSQQ,NPIATM),PCBDSZ(NBEADSQQ,NPIATM), &
                  EFPIOLD2(NBEADSQQ),EFPIMOV2(NBEADSQQ)
  real(chm_real)  EFPIHNEW2,EFPIHOLD2,EHARMAVE2,EPOTAVE2,ETOL_FPI2,ETNE_FPI2,PISUM2
  real(chm_real)  RXNCOR
  integer ::      eesize
  real(chm_real)  ISTRVEC(4)   ! Isotropic vector for Donor-Acceptor atoms
  real(chm_real),allocatable,dimension(:) :: EBETA,EBETA2,EED,EED2,EEDOLD,EEDOLD2   ! End2end distance for each beads configuration

  if (MCEQUIL > MCCONF) then
     eesize = MCEQUIL
  else
     eesize = MCCONF
  endif
  call chmalloc('qub.src','QPIMC','EBETA',EESIZE,crl=EBETA)
  call chmalloc('qub.src','QPIMC','EBETA2',EESIZE,crl=EBETA2)
  call chmalloc('qub.src','QPIMC','EED',EESIZE,crl=EED)
  call chmalloc('qub.src','QPIMC','EED2',EESIZE,crl=EED2)
  call chmalloc('qub.src','QPIMC','EEDOLD',EESIZE,crl=EEDOLD)
  call chmalloc('qub.src','QPIMC','EEDOLD2',EESIZE,crl=EEDOLD2)
  DELH = ZERO
  !
  !  centroid energy
  !
  EBETA(1:EESIZE) = ZERO
  EFPIOLD(1:NBEADSQQ) = ZERO
  EED(1:EESIZE) = ZERO
  EEDOLD(1:EESIZE) = ZERO
  IF (QFEP) THEN
     EBETA2(1:EESIZE) = ZERO
     EFPIOLD2(1:NBEADSQQ) = ZERO
     EED2(1:EESIZE) = ZERO
     EEDOLD2(1:EESIZE) = ZERO
  ENDIF
  IF (OSTAGE .AND. ISTR) CALL GISTRVEC(X,Y,Z,RXNA,RXNB,RXNC,ISTRVEC)   ! Prepare donor-acceptor vector for isotropic open chain PI
  ECL_XBAR = ZERO
  ! First call to energy must calculate all H elements and 2 electron
  ! integrals. If fast Fock matrix updating is used, subsequent calls
  ! will use centroid Fock matrix elements 
  CALL PIENER(ECL_XBAR,X,Y,Z,DX,DY,DZ,.TRUE.)
  if(prnlev.ge.2) WRITE(OUTU,'(A,F20.5)') ' QPIMC> Classical energy = ',ECL_XBAR
  !
  !   Store classical (centroid) coordinates
  !
  DO I = 1,NPIATM
     JJ = IPIATM(I)
     XOLD(I) = X(JJ)
     YOLD(I) = Y(JJ)
     ZOLD(I) = Z(JJ)
  ENDDO
  IF (QFEP) CALL MASSPERT(CBDSX,CBDSY,CBDSZ,PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,MSQRAT)
  !  If bisect or staging, all beads moved to enforce centroid constraint.
  IF (BISE .or. STAGE) THEN
     NBEADMV = NBEADSQQ     ! # of beads to recenter
     NBMV = 1              ! # of beads to move, in BQCP always 1
  ELSE
     NBEADMV = NBMOVE      ! # of beads to recenter
     NBMV = NBMOVE         ! # of beads to move
  ENDIF

  loopITYP: DO ITYP = 1,2
     IF(ITYP.EQ.1) THEN
        IF(PRNLEV.GT.4) CALL PRINTHDR(ITYP,QFEP,OUTU)
        NCONFS = MCEQUIL
     ELSE
        IF(PRNLEV.GT.4) CALL PRINTHDR(ITYP,QFEP,OUTU)
        NCONFS = MCCONF
        !
        !   Initiate ETOL_FPI for path-integral sum (PISUM) prior to sampling
        ETOL_FPI = ZERO
        ! Mass-perturbation
        IF (QFEP) ETOL_FPI2 = ZERO
        IF (OSTAGE) THEN   ! Update end2end distances from last step of equilibration
           EEDOLD(1) = EED(MCEQUIL)
           IF (QFEP) EEDOLD2(1) = EED2(MCEQUIL)
        ENDIF
        DO I = 1,NBEADSQQ
           DO J = 1,NPIATM
              JJ = IPIATM(J)
              X(JJ) = XOLD(J) + CBDSX(I,J)
              Y(JJ) = YOLD(J) + CBDSY(I,J)
              Z(JJ) = ZOLD(J) + CBDSZ(I,J)
           ENDDO
           CALL PIENER(EFPIOLD(I),X,Y,Z,DX,DY,DZ,.FALSE.)
           IF (TIAC .OR. CHAC) EFPIOLD(I)=EFACT(I)*EFPIOLD(I)+FFACT(I)*PITI(DX,DY,DZ)
           IF (OSTAGE .AND. (I.EQ.1 .OR. I.EQ.NBEADSQQ)) EFPIOLD(I) = EFPIOLD(I)/TWO ! Average of first and last bead for open chain PI
           ETOL_FPI = ETOL_FPI+EFPIOLD(I)
           IF(PRNLEV.GT.10) WRITE(OUTU,'(A,I7,2F20.5)') ' QPIMC>',I,EFPIOLD(I),ETOL_FPI
           ! Mass-perturbation
           IF (QFEP) THEN
              DO J = 1,NPIATM
                 JJ = IPIATM(J)
                 X(JJ) = XOLD(J) + PCBDSX(I,J)
                 Y(JJ) = YOLD(J) + PCBDSY(I,J)
                 Z(JJ) = ZOLD(J) + PCBDSZ(I,J)
              ENDDO
              CALL PIENER(EFPIOLD2(I),X,Y,Z,DX,DY,DZ,.FALSE.)
              IF (TIAC .OR. CHAC) EFPIOLD2(I)=EFACT(I)*EFPIOLD2(I)+FFACT(I)*PITI(DX,DY,DZ)
              IF (OSTAGE .AND. (I.EQ.1 .OR. I.EQ.NBEADSQQ)) EFPIOLD2(I) = EFPIOLD2(I)/TWO ! Average of first and last bead for open chain PI
              ETOL_FPI2 = ETOL_FPI2+EFPIOLD2(I)
           ENDIF
        ENDDO
        ETOL_FPI = ETOL_FPI/(DBLE(NBEADSQ)*CHFACN)
        ETNE_FPI = ETOL_FPI
        IF(PRNLEV.GT.10) WRITE(OUTU,'(A,F20.5)') ' QPIMC> Centroid E = ',ETNE_FPI
        ! Mass-perturbation
        IF (QFEP) THEN
           ETOL_FPI2 = ETOL_FPI2/(DBLE(NBEADSQ)*CHFACN)
           ETNE_FPI2 = ETOL_FPI2
           IF(PRNLEV.GT.10) WRITE(OUTU,'(A,F20.5)') ' QPIMC> Centroid E = ',ETNE_FPI2
        ENDIF
        ! If ITYP
     ENDIF  ! ITYP.EQ.1

     ! Initiate polymer ring energies prior to equi. and sampling
     EFPIHOLD = FPIHARM(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IPIATM,AMASS,FPIFAC,OSTAGE)
     IF(PRNLEV.GT.10) WRITE(OUTU,'(A,F15.5)') ' QPIMC> Initial spring energy = ',EFPIHOLD
     !
     ! Mass-perturbation
     IF (QFEP) THEN
        EFPIHOLD2 = FPIHARM(PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,IPIATM,WMAIN,FPIFAC,OSTAGE)
        IF(PRNLEV.GT.10) WRITE(OUTU,'(A,F15.5)') ' QPIMC> Initial spring energy = ',EFPIHOLD2
     ENDIF
     !CC 
     ! 
     PISUM = ZERO
     EHARMAVE = ZERO
     EPOTAVE = ZERO
     ! Mass-perturbation
     IF (QFEP) THEN
        PISUM2 = ZERO
        EHARMAVE2 = ZERO
        EPOTAVE2 = ZERO
     ENDIF
     ! MC settings
     XREPEAT = ONE
     XACCEPT  = ZERO
     XREJECT  = ZERO

     !   cDTM Loop over # of eq/av configs
     loopNCON: DO NCON = 1,NCONFS
        !
        !  Select NBMOVE FPI nuclei and update coordinates
        if (.not. (BISE.or.STAGE)) CALL MCATMS2(NBMV,IBMOVE,NBEADSQQ)
        !   Save old beads
        DO I = 1,NPIATM
           DO J = 1,NBEADMV
              KK = IBMOVE(J)
              XBOLD(J,I) = CBDSX(KK,I)
              YBOLD(J,I) = CBDSY(KK,I)
              ZBOLD(J,I) = CBDSZ(KK,I)
           ENDDO
        ENDDO
        !  Bisect, staging or random cubical MC move
        IF (BISE.OR.STAGE) THEN
           IF (BISE) CALL BISECT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IBMOVE,NBMV,LAMBDA,KLEV)
           IF (STAGE) THEN
              IF (OSTAGE) THEN
!!! NOTE: Mass-perturbation has to be worked out!!!
                 CALL OSTAGING(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,LAMBDA,NBEADSQQ,EED(NCON),ISTR,ISTRVEC)
                 EEDOLD(NCON+1) = EED(NCON)  ! Update previous end2end distance
                 IF (QFEP) EEDOLD2(NCON+1) = EED2(NCON)
              ELSE
                 CALL STAGING(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IBMOVE,NBMV,LAMBDA,NBEADSQQ)
              ENDIF
           ENDIF
           !  Enforce centroid constraint for all the particles in bisection or staging
           CALL FPICNT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,NBEADSQQ)
        ELSE
           CALL MCMV(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IBMOVE,NBMV,LAMBDA,LAMBDA2,PIMCMV)
           !  Enforce centroid constraint for the moved particles only for metropolis
           CALL FPICNT2(CBDSX,CBDSY,CBDSZ,XBOLD,YBOLD,ZBOLD,NPIATM,NBEADSQQ,NBMV,IBMOVE)
        ENDIF

        !  Get energy of polymer rings
        EFPIHNEW = FPIHARM(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IPIATM,AMASS,FPIFAC,OSTAGE)
        ! Mass perturbation - no MC check for free-particles will be done since QFEP uses BISE or STAGE
        IF (QFEP) THEN
           CALL MASSPERT(CBDSX,CBDSY,CBDSZ,PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,MSQRAT)
           EFPIHNEW2 = FPIHARM(PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,IPIATM,WMAIN,FPIFAC,OSTAGE)
        ENDIF

        !
        !   Get dH only if not sampling accurate free particle distribution (e.g. bisecting).
        IF (BISE .or. STAGE) THEN
           DELH = ZERO
        ELSE
           DELH = EFPIHNEW-EFPIHOLD
        ENDIF
        !         if(prnlev.ge.2) write(6,*) 'delh',delh
        BFACT = EXP(-DELH*BETA_FPI)
        XTMP = RANUMC()
        !         write(99,'(A,5F10.5)')'efpihnew,efpihold,delh,bfact,xtmp',efpihnew,efpihold,delh,bfact,xtmp

        !
        !   cDTM MC if
        !   Averaging over free beads
        IF(BFACT.GE.XTMP) THEN
           !
           !  Accept the old MC configuration, and update.
           !  New configuration is updated until the next acceptance move.
           !
           !  Don't do qm calculation during free-particle equilibration part
           IF(ITYP.EQ.2) THEN

              ETNE_FPI = ETOL_FPI
              IF (QFEP) ETNE_FPI2 = ETOL_FPI2
              DO J = 1,NBEADMV
                 KK = IBMOVE(J)
                 DO I = 1,NPIATM
                    JJ = IPIATM(I)
                    X(JJ) = XOLD(I)+CBDSX(KK,I)
                    Y(JJ) = YOLD(I)+CBDSY(KK,I)
                    Z(JJ) = ZOLD(I)+CBDSZ(KK,I)
                 ENDDO
                 !  get energy
                 CALL PIENER(EFPIMOV(J),X,Y,Z,DX,DY,DZ,.FALSE.)
                 !   cDTM Average over moved configs 
                 IF (TIAC .OR. CHAC) EFPIMOV(J)=EFACT(J)*EFPIMOV(J)+FFACT(J)*PITI(DX,DY,DZ)
                 IF (OSTAGE .AND. (J.EQ.1 .OR. J.EQ.NBEADSQQ)) EFPIMOV(J) = EFPIMOV(J)/TWO ! Average of first and last bead for open chain PI
                 ETNE_FPI = ETNE_FPI+(EFPIMOV(J)-EFPIOLD(KK))/(DBLE(NBEADSQ)*CHFACN)
                 !  if(prnlev.ge.2) write(6,'(A,2I,3F15.5)') 'mc',ncon,j,efpimov(j),EFPIOLD(KK),etne_fpi
                 ! Mass perturbation
                 IF (QFEP) THEN
                    DO I = 1,NPIATM
                       JJ = IPIATM(I)
                       X(JJ) = XOLD(I)+PCBDSX(KK,I)
                       Y(JJ) = YOLD(I)+PCBDSY(KK,I)
                       Z(JJ) = ZOLD(I)+PCBDSZ(KK,I)
                    ENDDO
                    CALL PIENER(EFPIMOV2(J),X,Y,Z,DX,DY,DZ,.FALSE.)
                    IF (TIAC .OR. CHAC) EFPIMOV2(J)=EFACT(J)*EFPIMOV2(J)+FFACT(J)*PITI(DX,DY,DZ)
                    IF (OSTAGE .AND. (J.EQ.1 .OR. J.EQ.NBEADSQQ)) EFPIMOV2(J) = EFPIMOV2(J)/TWO ! Average of first and last bead for open chain PI
                    ETNE_FPI2 = ETNE_FPI2+(EFPIMOV2(J)-EFPIOLD2(KK))/(DBLE(NBEADSQ)*CHFACN)
                 ENDIF
              ENDDO
              !   Sum up action
              EBETA(NCON) = EXP(-BETA_FPI*(ETOL_FPI-ECL_XBAR))
              PISUM = PISUM+XREPEAT*EBETA(NCON)
              EPOTAVE = EPOTAVE+XREPEAT*ETOL_FPI
              ! Mass perturbation
              IF (QFEP) THEN
                 EBETA2(NCON) = EXP(-BETA_FPI*(ETOL_FPI2-ECL_XBAR))
                 PISUM2 = PISUM2+XREPEAT*EBETA2(NCON)
                 EPOTAVE2 = EPOTAVE2+XREPEAT*ETOL_FPI2
              ENDIF
              !
              ! DTM mvd outside if
              !                XACCEPT = XACCEPT+XREPEAT
              ETOL_FPI = ETNE_FPI
              ! Mass perturbation
              IF (QFEP) ETOL_FPI2 = ETNE_FPI2
              DO J = 1,NBEADMV
                 KK = IBMOVE(J)
                 EFPIOLD(KK) = EFPIMOV(J)
                 IF (QFEP) EFPIOLD2(KK) = EFPIMOV2(J)
              ENDDO
           ENDIF    ! ITYP.EQ.2

           XACCEPT = XACCEPT+XREPEAT
           EHARMAVE = EHARMAVE+XREPEAT*EFPIHOLD
           EFPIHOLD = EFPIHNEW
           ! Mass perturbation
           IF (QFEP) THEN
              EHARMAVE2 = EHARMAVE2+XREPEAT*EFPIHOLD2
              EFPIHOLD2 = EFPIHNEW2
           ENDIF
           XREPEAT = ONE

           !   cDTM MC else
        ELSE      ! BFACT.GE.XTMP
           !
           ! Bring back the original coordinates
           ! Only if Metropolis - bisect and staging have 100% acceptance
           DO I = 1,NPIATM
              DO J = 1,NBEADMV
                 KK = IBMOVE(J)
                 CBDSX(KK,I) = XBOLD(J,I)
                 CBDSY(KK,I) = YBOLD(J,I)
                 CBDSZ(KK,I) = ZBOLD(J,I)
              ENDDO
           ENDDO
           XREJECT = XREJECT+ONE
           XREPEAT = XREPEAT+ONE
           !   cDTM MC endif
        ENDIF    ! BFACT.GE.XTMP
        !  DTM Moved outside if/else
        !            XACCEPT = XACCEPT + ONE
        !         write(98,*) 'accept:',ncon,int(xreject),int(xrepeat),int(xaccept)
        !
        ! Print potential and kinetic energy at each FP iteration
        IF(PRNLEV.GT.4 .AND. ITYP.EQ.2) THEN
           IF (QFEP) THEN
              WRITE(OUTU,'(A,I7,F20.5,F20.5,F12.5)') ' QPIMC> ',NCON,ETNE_FPI,ETNE_FPI2,EFPIHNEW2
           ELSE
              WRITE(OUTU,'(A,I7,F20.5,F12.5)') ' QPIMC> ',NCON,ETNE_FPI,EFPIHNEW
           ENDIF
        ENDIF
        ! cDTM end do NCONFS
     ENDDO loopNCON

     if(prnlev.ge.2) then
        WRITE(OUTU,'(A,I12)') ' QPIMC> NCONF: ',NCON-1
        WRITE(OUTU,'(A,I12)') ' QPIMC> NRJCT: ',INT(XREJECT)
        WRITE(OUTU,'(A,I12)') ' QPIMC> NACCP: ',INT(XACCEPT)
        IF(XACCEPT.GT.ZERO) WRITE(OUTU,'(A,F8.4)') ' QPIMC> ACC %: ',100.0D0*(NCON-XREJECT-1)/(NCON-1)
     end if
     !
     !  averaging
     !
     IF(XACCEPT.EQ.ZERO) XACCEPT = DBLE(NCONFS)
     !      if(prnlev.ge.2) write(6,*) 'accept:',int(xaccept)
     !
     EHARMAVE = EHARMAVE/XACCEPT
     ! Mass perturbation
     IF (QFEP) EHARMAVE2 = EHARMAVE2/XACCEPT
     !   cDTM end do ITYP
  ENDDO loopITYP

  ! Compute MC averages
  PIAVE = PISUM/XACCEPT
  EPOTAVE = EPOTAVE/XACCEPT
  !      if(prnlev.ge.2) WRITE(NQUBOUT,'(A,E18.8)') ' QPIMC> PIAVE = ',PIAVE
  if(prnlev.ge.2) WRITE(OUTU,'(A,E18.8)') ' QPIMC> PIAVE = ',PIAVE
  ! Mass perturbation
  IF (QFEP) THEN
     PIAVE2 = PISUM2/XACCEPT
     EPOTAVE2 = EPOTAVE2/XACCEPT
     if(prnlev.ge.2) WRITE(OUTU,'(A,E18.8)') ' QPIMC> PIAVE = ',PIAVE2
  ENDIF

  !   Restore original coordinates
  DO I = 1,NPIATM
     JJ = IPIATM(I)
     X(JJ) = XOLD(I)
     Y(JJ) = YOLD(I)
     Z(JJ) = ZOLD(I)
  ENDDO
  ! Write open chain PI simulation data - weighting factor and end2end bead distance
  IF (OSTAGE) THEN
     RXNCOR = GETRXN(X,Y,Z,RXNA,RXNB,RXNC)
     WRITE(NMOMDIS,'(I7,F18.5)') NCONFS,RXNCOR
     IF (QFEP) WRITE(NMOMDIS2,'(I7,F18.5)') NCONFS,RXNCOR
     DO I = 1,NCONFS
        WRITE(NMOMDIS,'(E18.8,F12.5)') EBETA(I),EEDOLD(I)
        IF (QFEP) WRITE(NMOMDIS2,'(E18.8,F12.5)') EBETA2(I),EEDOLD2(I)
     ENDDO
  ENDIF
 
  !
  !  write out all beads coordinates
  !
  IF(NBOUT_UNIT.GT.0) THEN
     WRITE (NBOUT_UNIT,*) NPIATM,NBEADSQQ
     DO I = 1,NPIATM
        WRITE (NBOUT_UNIT,'(6F12.8)') (CBDSX(J,I),CBDSY(J,I),CBDSZ(J,I),J=1,NBEADSQQ)
     ENDDO
  ELSE
     !         CALL WRNDIE(10,'FPIMC>','final bead coordinates not written')
     if(prnlev.ge.2) WRITE(OUTU,'(A)') ' QPIMC> Final bead coordinates not written'
  ENDIF
  call chmdealloc('qub.src','QPIMC','EBETA',EESIZE,crl=EBETA)
  call chmdealloc('qub.src','QPIMC','EBETA2',EESIZE,crl=EBETA2)
  call chmdealloc('qub.src','QPIMC','EED',EESIZE,crl=EED)
  call chmdealloc('qub.src','QPIMC','EED2',EESIZE,crl=EED2)
  call chmdealloc('qub.src','QPIMC','EEDOLD',EESIZE,crl=EEDOLD)
  call chmdealloc('qub.src','QPIMC','EEDOLD2',EESIZE,crl=EEDOLD2)
  !
  RETURN
END SUBROUTINE QPIMC

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Get energy required for PI evaluation
SUBROUTINE PIENER(E,X,Y,Z,DX,DY,DZ,CLAS)
  use chm_kinds
  use chm_types

  use dimens_fcm
  use psf
  ! next 2 added for energy call
  use bases_fcm
  use energym
  !
  use number
#if KEY_QUANTUM==1
  use quantm  
#endif
  use qubpi

#if KEY_SQUANTM==1
  use squantm 
#if KEY_PARALLEL==1
  use parallel 
#endif
#endif 
  implicit none

  real(chm_real) :: X(:), Y(:), Z(:), DX(:), DY(:), DZ(:)
  real(chm_real)   E
  LOGICAL  CLAS
  !
  real(chm_real)   EVDW
  INTEGER  QNUMNOD,QMYNOD

  E = ZERO
  EVDW = ZERO
! Only TI or CH use gradients
  IF (TIAC .OR. CHAC) THEN
! namkh: I made entire gradient arrays to zero, otherwise
!        the numbers will accumulate.(just for code safety.)
     dx(1:natom)=zero
     dy(1:natom)=zero
     dz(1:natom)=zero
  ENDIF
#if KEY_SCCDFTB==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
  E = EPROP(EPOT)
#if KEY_PARALLEL==1
  CALL VDGBR(DX,DY,DZ,1)
#endif 
#endif 

#if KEY_QUANTUM==1
#if KEY_PARALLEL==1
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
  E = EPROP(EPOT)
  CALL VDGBR(DX,DY,DZ,1)
#else /**/
  IF (TIAC .OR. CHAC) THEN
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
     E = EPROP(EPOT)
  ELSE
     IF (FFOCK .OR. FENER) THEN
    ! Use tailored EAMPAC routine that avoids computing derivatives
    ! and can use fast Fock matrix updating
        CALL QUBEAMPAC(E,X,Y,Z,FFOCK,CLAS,IQMPIATM)
     ELSE
        CALL EAMPAC(E,X,Y,Z,DX,DY,DZ)
     ENDIF
     IF (NATOM.GT.NATQM .AND. QETERM(QMVDW)) CALL EVDWQM(EVDW,X,Y,Z,DX,DY,DZ)
  ENDIF
#endif 
#endif 

! SQUANTM has specialized routines for fast parallel path-integral evaluations
#if KEY_SQUANTM==1
#if KEY_PARALLEL==1
  QNUMNOD = NUMNOD
  NUMNOD = 1
  QMYNOD = MYNOD
  MYNOD = 0 
#endif 
  CALL SQMMME(E,X,Y,Z,DX,DY,DZ)
  IF (NATOM.GT.NATQM(1) .AND. QETERM(QMVDW)) CALL EVDWQM(EVDW,X,Y,Z,DX,DY,DZ)
#if KEY_PARALLEL==1
  NUMNOD = QNUMNOD  
  MYNOD = QMYNOD    
#endif 
#endif 

  E = E + EVDW

  RETURN
END SUBROUTINE PIENER

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Get energy required for PI evaluation
FUNCTION PITI(DX,DY,DZ) result(PITI_rt)
  use chm_kinds
  use number
  use qubpi
  use parallel

  implicit none
  ! Passed arguments
  real(chm_real) DX(*),DY(*),DZ(*)
  ! Local variables
  INTEGER :: J,JJ
  real(chm_real) HOAC,PITI_rt   ! Higher-order action

  HOAC = ZERO
  DO J = 1,NPIATM
     JJ = IPIATM(J)
     HOAC = HOAC + HOFAC(J)*(DX(JJ)**2 + DY(JJ)**2 + DZ(JJ)**2)
!         write(6,'(3I,5E15.5)')mynod,j,jj,HOFAC(J),DX(JJ),DY(JJ),DZ(JJ),HOAC
  ENDDO
  PITI_rt = HOAC

  RETURN
END FUNCTION PITI

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION FPIHARM(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,IPIATM,AMASS,FPIFAC,OSTAGE)
  use chm_kinds
  use number
  implicit none

  INTEGER :: NPIATM,NBEADSQ,IPIATM(NPIATM)
  real(chm_real) AMASS(*),FPIFAC,fpiharm
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM)
  INTEGER :: I,J,II
  real(chm_real) AM,UHARM,RR,temp1
  LOGICAL :: OSTAGE
  !
  UHARM = ZERO
  FPIHARM = ZERO
  DO I = 1,NPIATM
     !====         II = IPIATM(I)
     !====         AM = AMASS(II)
     temp1=FPIFAC*AMASS(IPIATM(I))
     DO J = 1,NBEADSQ-1
        RR=((CBDSX(J,I)-CBDSX(J+1,I))**2+(CBDSY(J,I)-CBDSY(J+1,I))**2+(CBDSZ(J,I)-CBDSZ(J+1,I))**2)
        UHARM = UHARM+temp1*RR
     ENDDO
     IF (.NOT. OSTAGE) THEN   ! Open chain polymer
       RR=((CBDSX(1,I)-CBDSX(NBEADSQ,I))**2+(CBDSY(1,I)-CBDSY(NBEADSQ,I))**2+(CBDSZ(1,I)-CBDSZ(NBEADSQ,I))**2)
       UHARM = UHARM+temp1*RR
     ENDIF
  ENDDO
  FPIHARM = UHARM
  !
  RETURN
END FUNCTION FPIHARM

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE MCATMS(IBMOVE,NBEADSQ)
  use chm_kinds
  implicit none

  INTEGER :: NBEADSQ,IBMOVE(NBEADSQ)
  !
  INTEGER :: I

  DO I = 1,NBEADSQ
     IBMOVE(I) = I
  ENDDO

  RETURN
END SUBROUTINE MCATMS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE MCATMS2(NBMOVE,IBMOVE,NBEADSQ)
  use chm_kinds
  implicit none

  INTEGER :: NBMOVE,NBEADSQ,IBMOVE(NBEADSQ)
  INTEGER :: I,J,K

  DO I = 1,NBMOVE
10   K = INT(DBLE(NBEADSQ)*RANUMC())+1
     DO J = 1,I-1
        IF(K.EQ.IBMOVE(J)) GO TO 10
     ENDDO
     IBMOVE(I) = K

  ENDDO
  RETURN
END SUBROUTINE MCATMS2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE PRINTHDR(ITYP,QFEP,OUTU)
  use chm_kinds
  implicit none
  !
  INTEGER :: ITYP,OUTU
  LOGICAL QFEP

  IF (ITYP.EQ.1) THEN
     WRITE(OUTU,'(A)') ' ======================================'
     WRITE(OUTU,'(A)') ' QPIMC> FP equilibration stage'
     WRITE(OUTU,'(A)') ' ======================================'
  ELSE 
     WRITE(OUTU,'(A)') ' ======================================'
     WRITE(OUTU,'(A)') ' QPIMC> FP sampling stage'
     WRITE(OUTU,'(A)') ' ======================================'
     IF (QFEP) THEN
        WRITE(OUTU,'(A)') ' QPIMC     Step     POTENE1     POTENE2      KINENE'
     ELSE
        WRITE(OUTU,'(A)') ' QPIMC     Step      POTENE      KINENE'
     ENDIF
  ENDIF

  RETURN
END SUBROUTINE PRINTHDR

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION GETRXN(X,Y,Z,RXNA,RXNB,RXNC) result(getrxn_rt)
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),getrxn_rt
  INTEGER RXNA,RXNB,RXNC
  !
  real(chm_real) XAB,YAB,ZAB,XBC,YBC,ZBC,RAB,RBC

  XAB = X(RXNB) - X(RXNA)
  YAB = Y(RXNB) - Y(RXNA)
  ZAB = Z(RXNB) - Z(RXNA)
  RAB = SQRT(XAB*XAB + YAB*YAB + ZAB*ZAB)
  ! If RXNC <= 0; reaction coordinate simple diatomic distance
  IF (RXNC.GT.0 .AND. RXNC.LT.99999) THEN
     XBC = X(RXNC) - X(RXNB)
     YBC = Y(RXNC) - Y(RXNB)
     ZBC = Z(RXNC) - Z(RXNB)
     RBC = SQRT(XBC*XBC + YBC*YBC + ZBC*ZBC)
  ELSE
     RBC = 0
  ENDIF
  getrxn_rt = RAB - RBC

  RETURN
END FUNCTION GETRXN

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Sort PI results into bins along reaction coordinate
SUBROUTINE QUBIN(RXNCOOR,RXNPIAV,CLNUM)
  use chm_kinds
  use stream
  use sizes
  use number, only : zero, one, two
  use qubpi
  implicit none
  !
  INTEGER :: CLNUM 
  real(chm_real)  RXNCOOR(CLNUM),RXNPIAV(CLNUM)
  !
  INTEGER :: I,BINCOOR
  real(chm_real)  GRID,PIBAVE,PIBSTD,FPIBAVE,FPIBSTD,LOW,HIGH
  real(chm_real)  BIN(-MAXBIN:MAXBIN),PIBIN(-MAXBIN:MAXBIN),PIBIN2(-MAXBIN:MAXBIN)

  GRID = one / BINWIDTH
  BIN(-MAXBIN:MAXBIN)    = zero
  PIBIN(-MAXBIN:MAXBIN)  = zero
  PIBIN2(-MAXBIN:MAXBIN) = zero
  DO I=1,CLNUM
     BINCOOR        = INT(GRID*(RXNCOOR(I) + RESLN))
     BIN(BINCOOR)   = BIN(BINCOOR) + one
     PIBIN(BINCOOR) = PIBIN(BINCOOR) + RXNPIAV(I)
     PIBIN2(BINCOOR)= PIBIN2(BINCOOR)+ RXNPIAV(I)**2
     !            write(*,*)' QUBIN> ',RXNCOOR(i),RXNPIAV(i)
  ENDDO
  DO I=-MAXBIN,MAXBIN,1
     IF (BIN(I).GT.zero) THEN
        PIBAVE  = PIBIN(I) / BIN(I)
        FPIBAVE =-LOG(PIBAVE)/BETA_FPI
        IF (BIN(I).EQ.one) THEN
           FPIBSTD = zero
        ELSE   
           PIBSTD = SQRT((PIBIN2(I)-PIBIN(I)**2/BIN(I))/BIN(I))
           !               FPIBSTD = -LOG(PIBSTD)/BETA_FPI
           ! Standard deviation of log(x)
           FPIBSTD = FPIBSTD/(PIBAVE*BETA_FPI)
        ENDIF
        IF (I .LE. 0) THEN
           LOW  = DBLE(I)*BINWIDTH - BINWIDTH - RESLN
           HIGH = DBLE(I)*BINWIDTH - RESLN
        ELSE
           LOW  = DBLE(I)*BINWIDTH - RESLN
           HIGH = DBLE(I)*BINWIDTH + BINWIDTH - RESLN
        ENDIF
        !            WRITE(NQUBOUT,99) LOW,HIGH,FPIBAVE,FPIBSTD,INT(BIN(I))
        WRITE(OUTU,99) LOW,HIGH,FPIBAVE,FPIBSTD,INT(BIN(I))
     ENDIF
  ENDDO

99 FORMAT(1X,'Z = ',F8.4,1X,'-',1X,F8.4,1X,'Del G(CL->QM)  = ',E12.5,1X,'+/-',1X,E12.5,1x,'Bins = ',I7)
  RETURN
END SUBROUTINE QUBIN

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Get isotropic donor-acceptor vector
SUBROUTINE GISTRVEC(X,Y,Z,RXNA,RXNB,RXNC,ISTRVEC)
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),ISTRVEC(4)
  INTEGER RXNA,RXNB,RXNC
  !
  ! If RXNC <= 0; reaction coordinate simple diatomic distance
!  IF (RXNC.GT.0 .AND. RXNC.LT.99999) THEN
     ISTRVEC(1) = X(RXNC) - X(RXNA)
     ISTRVEC(2) = Y(RXNC) - Y(RXNA)
     ISTRVEC(3) = Z(RXNC) - Z(RXNA)
!  ELSE
!     ISTRVEC(1) = X(RXNC) - X(RXNA)
!     ISTRVEC(2) = Y(RXNC) - Y(RXNA)
!     ISTRVEC(3) = Z(RXNC) - Z(RXNA)
!  ENDIF
  ISTRVEC(4) = SQRT(ISTRVEC(1)*ISTRVEC(1) + ISTRVEC(2)*ISTRVEC(2) + ISTRVEC(3)*ISTRVEC(3))

  RETURN
END SUBROUTINE GISTRVEC


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION RANUMC () result(RANUMC_rt)
  use chm_kinds
  use sizes
  use number, only : zero, half, one, two
  use qubpi
  implicit none
  !
  !-----------------------------------------------------------------------
  !
  !     RANDOM NUMBER GENERATOR
  !     SEPTEMBER, 1986: WLJ & J. M. Briggs
  !
  !     NOTES ON USAGE:
  !     THE STATEMENT "COMMON/RANDS/IRN" MUST APPEAR IN EACH
  !     PROGRAM,SUBROUTINE... WHERE THE SEED VALUE (IRN) IS
  !     BEING WRITTEN OUT OR BEING READ IN. THIS FUNCTION IS
  !     USED IN THE SAME MANNER AS THAT USED ON THE GOULD,
  !     (I.E. "RANDN = RANU()" ). IRN SHOULD INITIALLY BE
  !     AN ODD, I6 INTEGER BUT CAN GET AS BIG AS I7. JMB
  !     IMOD-1 UNIQUE RANDOM NUMBERS ARE GENERATED . A SINGLE
  !     PRECISION LINEAR CONGRUENTIAL GENERATOR IS USED WITH
  !     SHUFFLING OF THE CONSTANT AND OF WHEN THE SHUFFLING IS
  !     DONE. CONSEQUENTLY, THE PERIOD IS EXTREMELY LONG - NONE
  !     WAS FOUND IN TESTS GENERATING MILLIONS OF RANDOM NUMBERS.
  !-----------------------------------------------------------------------
  !
  real(chm_real)  RNJ,FAC,ranumc_rt
  ! namkh: 12-21-2009
  ! I think it should be saved, otherwise, whenever it calls RANUMC, it starts with
  ! same ICNT, ICON, etc.
  ! but, have to check with DTM.
  integer,save :: ICNT=0   ,ICHG=1167     ,ICHG0=1167  ,ICN0=458753759,  &
       IMUL=1173,ICON=458753759,IMOD=1048573, &
       JMUL=1161,JCON=458716759,JMOD=1048573,JRN=124690
  !
  ICNT = ICNT+1
  IF (ICNT.EQ.ICHG) THEN
     !
     !-----------------------------------------------------------------------
     !
     !     CHANGE ICON USING SECONDARY GENERATOR
     !
     !-----------------------------------------------------------------------
     !
     JRN = JRN*JMUL+JCON
     JRN = MOD(JRN,JMOD)
     RNJ = FLOAT(JRN)/FLOAT(JMOD)
     IF (RNJ.GT.half) THEN
        FAC = one + half*RNJ
     ELSE
        FAC = one - half*RNJ
     ENDIF
     FAC = FLOAT(ICN0)*FAC
     ICON = INT(FAC)
     !
     !-----------------------------------------------------------------------
     !
     !     CHANGE ICHG USING SECONDARY GENERATOR
     !
     !-----------------------------------------------------------------------
     !
     JRN = JRN*JMUL+JCON
     JRN = MOD(JRN,JMOD)
     RNJ = FLOAT(JRN)/FLOAT(JMOD)
     IF (RNJ.GT.half) THEN
        FAC = one + half*RNJ
     ELSE
        FAC = one - half*RNJ
     ENDIF
     FAC  = FLOAT(ICHG0)*FAC
     ICHG = INT(FAC)
     ICNT = 0
  ENDIF
  !
  !-----------------------------------------------------------------------
  !
  !     GENERATE RANDOM NUMBER
  !
  !-----------------------------------------------------------------------
  !
  IRN = IRN*IMUL+ICON
  IRN = MOD(IRN,IMOD)
  RANUMC_rt = DBLE(IRN)/DBLE(IMOD)
  !
  RETURN
END FUNCTION RANUMC

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION GAUSDEV(SIGMA,XMID)
  !   DTM From Numerical Recipies, Ch. 7.2 
  !   Box-Muller algorithm
  use chm_kinds
  use sizes
  use number,only : zero, one, two
  use qubpi
  implicit none

  real(chm_real) SIGMA,XMID,V1,V2,R,FAC,gausdev
  INTEGER, save :: ISET=0    ! initialized..
  real(chm_real), save :: GSET

  IF (ISET.EQ.0) THEN
     do
        V1=two*RANUMC() - one
        V2=two*RANUMC() - one
        R =V1*V1 + V2*V2
        if(R.GT.zero .and. R.lt.one) EXIT
     end do
     FAC=SQRT(-two*LOG(R)/R)
     GSET=V1*FAC
     GAUSDEV=V2*FAC*SIGMA + XMID
     ISET=1
  ELSE
     GAUSDEV=GSET*SIGMA + XMID
     ISET=0
  ENDIF
  RETURN
END FUNCTION GAUSDEV

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION GAUSDEV2(SIGMA,XMID)
  !   DTM From Numerical Recipies, Ch. 7.2
  !   Joseph Leva's algorithm
  !   "A fast normal random number generator" in ACM Trans. Math. Softw. 1992
  !   Algorithm has double precision
  use chm_kinds
  use sizes
!  use number,only : zero, one, two
  use qubpi
  implicit none

  real(chm_real) :: sigma,xmid
  real(chm_real) :: u,v,q,gausdev2

  call gausaux(u,v,q)
  do while (q > 0.27597 .and. (q > 0.27846 .or. sqrt(v) > -4.0d0*log(u)*sqrt(u)))
     call gausaux(u,v,q)
  enddo
  gausdev2 = xmid + sigma*v/u

  return

END FUNCTION GAUSDEV2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE GAUSAUX(U,V,Q)
  ! Auxiliary routine for Leva's method
  use chm_kinds
  implicit none
  real(chm_real) :: u,v,q
  real(chm_real) :: x,y

  u = ranumc()
  v = 1.7156d0*(ranumc() - 0.5d0)
  x = u - 0.449871d0
  y = dabs(v) + 0.386595d0
  q = sqrt(x) + y*(0.19600d0*y - 0.25472d0*x)

  return

END SUBROUTINE GAUSAUX

#endif /* (qmsetmain)*/

!  Parallel (MPI) additions for the QUB path-integral program
!  using the QCP path-integral method
!  Written by Dan T. Major 01/XX/2005-24/12/2010
!  Contact: majort@mail.biu.ac.il
!
#if KEY_QUANTUM==1 || KEY_SQUANTM==1 || KEY_SCCDFTB==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 /*quant*/
#if KEY_PARALLEL==1 /*pll*/
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE QMPIMC(X,Y,Z,DX,DY,DZ,CBDSX,CBDSY,CBDSZ,AMASS,WMAIN,EFACT,FFACT,NATOM,IBMOVE,PIAVE,PIAVE2)
  use chm_kinds
  use memory
  use stream
  use number
  use dimens_fcm
  use sizes
  use qubpi
  !
  use parallel
  implicit none
  !
  real(chm_real) :: X(:), Y(:), Z(:), DX(:), DY(:), DZ(:)
  real(chm_real)   CBDSX(NBEADSQQ,NPIATM),CBDSY(NBEADSQQ,NPIATM),CBDSZ(NBEADSQQ,NPIATM),AMASS(*),WMAIN(*)
  real(chm_real)   EFACT(NBEADSQQ),FFACT(NBEADSQQ)
  real(chm_real)   PIAVE,PIAVE2
  INTEGER ::       IBMOVE(NBEADSQQ),NATOM
  !
  INTEGER I,J
  INTEGER JJ,KK,ITYP,NCONFS,NCON,NBEADMV,NBMV
  INTEGER ISTA,IEND,JSTA,JEND,KSTA,KEND,BUFSIZ
  real(chm_real)  PISUM,XACCEPT,XREJECT,XREPEAT
  real(chm_real)  ETOL_FPI,ETNE_FPI
  real(chm_real)  ECL_XBAR,EFPIHOLD
  real(chm_real)  EFPIHNEW,DELH,BFACT,XTMP
  real(chm_real)  XOLD(NPIATM),YOLD(NPIATM),ZOLD(NPIATM),EFPIOLD(NBEADSQQ),EFPIMOV(NBEADSQQ)
  real(chm_real)  XBOLD(NBEADSQQ,NPIATM),YBOLD(NBEADSQQ,NPIATM),ZBOLD(NBEADSQQ,NPIATM)
  real(chm_real)  EHARMAVE,EPOTAVE
  ! Mass-perturbation variables
  real(chm_real)  PCBDSX(NBEADSQQ,NPIATM),PCBDSY(NBEADSQQ,NPIATM),PCBDSZ(NBEADSQQ,NPIATM), &
       EFPIOLD2(NBEADSQQ),EFPIMOV2(NBEADSQQ)
  real(chm_real)  EFPIHNEW2,EFPIHOLD2,EHARMAVE2,EPOTAVE2,ETOL_FPI2,ETNE_FPI2,PISUM2
  real(chm_real)  RXNCOR
  real(chm_real)  ETOL_FPI_ALL,ETNE_FPI_ALL,ETOL_FPI_ALL2,ETNE_FPI_ALL2
  integer ::      eesize
  real(chm_real)  ISTRVEC(4)   ! Isotropic vector for Donor-Acceptor atoms
  real(chm_real),allocatable,dimension(:) :: EBETA,EBETA2,EED,EED2,EEDOLD,EEDOLD2   ! End2end distance for each beads configuration

  if (MCEQUIL > MCCONF) then
     eesize = MCEQUIL
  else
     eesize = MCCONF
  endif
  call chmalloc('qub.src','QMPIMC','EBETA',EESIZE,crl=EBETA)
  call chmalloc('qub.src','QMPIMC','EBETA2',EESIZE,crl=EBETA2)
  call chmalloc('qub.src','QMPIMC','EED',EESIZE,crl=EED)
  call chmalloc('qub.src','QMPIMC','EED2',EESIZE,crl=EED2)
  call chmalloc('qub.src','QMPIMC','EEDOLD',EESIZE,crl=EEDOLD)
  call chmalloc('qub.src','QMPIMC','EEDOLD2',EESIZE,crl=EEDOLD2)
  !
  DELH = ZERO
  ! Let each processor take a sub-set of the beads (decided by PARA_RANGE
  ! function) then let node 0 take care of summing up total energy and
  ! compute path-integral.
  ! Node 0 takes care of generating bead configurations and sending
  ! coordinates to other nodes. All nodes calculate free-particle part
  ! (very cheap) One call to PARA_RANGE should do it
  CALL PARA_RANGE(1,NBEADSQQ,NUMNOD,MYNOD,ISTA,IEND)
  JSTA = ISTA
  JEND = IEND
  ! For normal Metropolis MC NBMOVE should be used and not NBEADSQ
  IF (.NOT.(BISE .OR. STAGE)) CALL PARA_RANGE(1,NBMOVE,NUMNOD,MYNOD,ISTA,IEND)
  BUFSIZ = NBEADSQQ*NPIATM
  ! Prepare array for vector allgather calls. Energy vectors (EFPIOLD/EFPIOLD2) must
  ! have same internal node distribution division as bead over node distribution
  ! The energy vectors are updated below using the VDGBRE routine
  DO I = 0,NUMNOD
     CALL PARA_RANGE(1,NBEADSQQ,NUMNOD,I,KSTA,KEND)
     JPARPT(I) = KSTA - 1
  ENDDO
#if KEY_QUANTUM==1 || KEY_SCCDFTB==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1
  ISTA=1
  JSTA=1
  KSTA=1
  IEND=NBEADSQQ
  JEND=NBEADSQQ
  KEND=NBEADSQQ
#endif 
  !
  !  centroid energy
  !
  EBETA(1:EESIZE) = ZERO
  EFPIOLD(1:NBEADSQQ) = ZERO
  EED(1:EESIZE) = ZERO
  EEDOLD(1:EESIZE) = ZERO
  IF (QFEP) THEN
     EBETA2(1:EESIZE) = ZERO
     EFPIOLD2(1:NBEADSQQ) = ZERO
     EED2(1:EESIZE) = ZERO
     EEDOLD2(1:EESIZE) = ZERO
  ENDIF
  IF (OSTAGE .AND. ISTR) CALL GISTRVEC(X,Y,Z,RXNA,RXNB,RXNC,ISTRVEC)
  ECL_XBAR = ZERO
  ! First call to energy must calculate all H elements and 2 electron
  ! integrals. If fast Fock matrix updating is used (old mopac code), subsequent calls
  ! will use centroid Fock matrix elements. All processors must call PIENER
  ! at centroid position to save necessary Fock matrix elements
  !      IF(MYNOD.EQ.0) THEN
  CALL PIENER(ECL_XBAR,X,Y,Z,DX,DY,DZ,.TRUE.)

  IF(MYNOD.EQ.0) WRITE(OUTU,'(A,F20.5)') ' QPIMC> Classical energy = ',ECL_XBAR
  !
  !   Store classical (centroid) coordinates
  !
  DO I = 1,NPIATM
     JJ = IPIATM(I)
     XOLD(I) = X(JJ)
     YOLD(I) = Y(JJ)
     ZOLD(I) = Z(JJ)
  ENDDO
  IF (QFEP) CALL MASSPERT(CBDSX,CBDSY,CBDSZ,PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,MSQRAT)
  !  If bisect or staging, all beads moved to enforce centroid constraint.
  IF (BISE .or. STAGE) THEN
     NBEADMV = NBEADSQQ    ! # of beads to recenter
     NBMV = 1              ! # of beads to move, in BQCP always 1
  ELSE
     NBEADMV = NBMOVE      ! # of beads to recenter
     NBMV = NBMOVE         ! # of beads to move
  ENDIF

  ! Only do equilibration first iteration
  loopITYP: DO ITYP = 1,2
     IF(ITYP.EQ.1) THEN
        IF(MYNOD.EQ.0.AND.PRNLEV.GT.4) CALL PRINTHDR(ITYP,QFEP,OUTU)
        NCONFS = MCEQUIL
     ELSE
        IF(MYNOD.EQ.0.AND.PRNLEV.GT.4) CALL PRINTHDR(ITYP,QFEP,OUTU)
        NCONFS = MCCONF
        ! Initiate ETOL_FPI for path-integral sum (PISUM) prior to sampling
        ETOL_FPI = ZERO
        ! Mass-perturbation
        IF (QFEP) ETOL_FPI2 = ZERO
        IF (OSTAGE) THEN   ! Update end2end distances from last step of equilibration
           EEDOLD(1) = EED(MCEQUIL)
           IF (QFEP) EEDOLD2(1) = EED2(MCEQUIL)
        ENDIF
        DO I = JSTA,JEND   ! When initializing MC run, loop over all beads even with Metropolis
           DO J = 1,NPIATM
              JJ = IPIATM(J)
              X(JJ) = XOLD(J) + CBDSX(I,J)
              Y(JJ) = YOLD(J) + CBDSY(I,J)
              Z(JJ) = ZOLD(J) + CBDSZ(I,J)
           ENDDO
           CALL PIENER(EFPIOLD(I),X,Y,Z,DX,DY,DZ,.FALSE.)
           IF (TIAC .OR. CHAC) EFPIOLD(I)=EFACT(I)*EFPIOLD(I)+FFACT(I)*PITI(DX,DY,DZ)
           IF (OSTAGE .AND. (I.EQ.1 .OR. I.EQ.NBEADSQQ)) EFPIOLD(I) = EFPIOLD(I)/TWO ! Average of first and last bead for open chain PI
           ETOL_FPI = ETOL_FPI+EFPIOLD(I)
           IF(PRNLEV.GT.10) WRITE(OUTU,'(A,I7,2F20.5)') ' QPIMC>',I,EFPIOLD(I),ETOL_FPI
           ! Mass-perturbation
           IF (QFEP) THEN
              DO J = 1,NPIATM
                 JJ = IPIATM(J)
                 X(JJ) = XOLD(J) + PCBDSX(I,J)
                 Y(JJ) = YOLD(J) + PCBDSY(I,J)
                 Z(JJ) = ZOLD(J) + PCBDSZ(I,J)
              ENDDO
              CALL PIENER(EFPIOLD2(I),X,Y,Z,DX,DY,DZ,.FALSE.)
              IF (TIAC .OR. CHAC) EFPIOLD2(I)=EFACT(I)*EFPIOLD2(I)+FFACT(I)*PITI(DX,DY,DZ)
              IF (OSTAGE .AND. (I.EQ.1 .OR. I.EQ.NBEADSQQ)) EFPIOLD2(I) = EFPIOLD2(I)/TWO ! Average of first and last bead for open chain PI
              ETOL_FPI2 = ETOL_FPI2+EFPIOLD2(I)
           ENDIF
        ENDDO
        ETOL_FPI_ALL = ETOL_FPI
#if KEY_SQUANTM==1
        if(numnod.gt.1) then
           CALL GCOMB(ETOL_FPI_ALL,1)
           CALL VDGBRE(EFPIOLD,JPARPT)  ! Global vector broadcast of energy array to all nodes
        end if
#endif 
        ETOL_FPI = ETOL_FPI/(DBLE(NBEADSQ)*CHFACN)
        ETOL_FPI_ALL = ETOL_FPI_ALL/(DBLE(NBEADSQ)*CHFACN)
        ETNE_FPI = ETOL_FPI
        ETNE_FPI_ALL = ETOL_FPI_ALL
        IF (MYNOD.EQ.0 .AND. PRNLEV.GT.10) WRITE(OUTU,'(A,F20.5)') ' QPIMC> Centroid E = ',ETNE_FPI_ALL
        ! Mass-perturbation
        IF (QFEP) THEN
           ETOL_FPI_ALL2 = ETOL_FPI2
#if KEY_SQUANTM==1
           if(numnod.gt.1) then
              CALL GCOMB(ETOL_FPI_ALL2,1)
              CALL VDGBRE(EFPIOLD2,JPARPT)
           end if
#endif 
           ETOL_FPI2 = ETOL_FPI2/(DBLE(NBEADSQ)*CHFACN)
           ETOL_FPI_ALL2 = ETOL_FPI_ALL2/(DBLE(NBEADSQ)*CHFACN)
           ETNE_FPI2 = ETOL_FPI2
           ETNE_FPI_ALL2 = ETOL_FPI_ALL2
           IF(MYNOD.EQ.0 .AND. PRNLEV.GT.10) WRITE(OUTU,'(A,F20.5)') ' QPIMC> Centroid E = ',ETNE_FPI_ALL2
        ENDIF
     ENDIF  ! ITYP.EQ.1
     ! Initiate polymer ring energies prior to equi. and sampling
     EFPIHOLD = FPIHARM(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IPIATM,AMASS,FPIFAC,OSTAGE)
     !
     ! Mass-perturbation
     IF (QFEP) EFPIHOLD2 = FPIHARM(PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,IPIATM,WMAIN,FPIFAC,OSTAGE)
     IF (MYNOD.EQ.0) THEN
        IF(PRNLEV.GT.10) WRITE(OUTU,'(A,F15.5)') ' QPIMC> Initial spring energy = ',EFPIHOLD
        IF (QFEP .AND. PRNLEV.GT.10)WRITE(OUTU,'(A,F15.5)')' QPIMC> Initial spring energy = ',EFPIHOLD2
        PISUM = ZERO
        EPOTAVE = ZERO
        EHARMAVE = ZERO
        ! Mass-perturbation
        IF (QFEP) THEN
           PISUM2 = ZERO
           EHARMAVE2 = ZERO
           EPOTAVE2 = ZERO
        ENDIF
        XREPEAT = ONE
        XACCEPT  = ZERO
        XREJECT  = ZERO
     ENDIF
     !   cDTM Loop over # of eq/av configs

     loopNCON: DO NCON = 1,NCONFS
        !
        !  Select NBMOVE FPI nuclei and update coordinates
        !
        IF (MYNOD.EQ.0) THEN
           if (.not. (BISE .or. STAGE)) then
              if (numnod == 1) then
                 CALL MCATMS2(NBMV,IBMOVE,NBEADSQQ)
              else
                 ! Each node will deal with its beads in its array range
                 DO I=1,NBMV
                    IBMOVE(1) = 0
                    CALL MCATMS2(1,IBMOVE,JPARPT(I)-JPARPT(I-1)) ! Fills IBMOVE(1) w/random #
                    IBMOVE(I) = IBMOVE(1) + JPARPT(I-1)
                 ENDDO
              endif
           endif
           !   Save old beads
           DO I = 1,NPIATM
              DO J = 1,NBEADMV
                 KK = IBMOVE(J)
                 XBOLD(J,I) = CBDSX(KK,I)
                 YBOLD(J,I) = CBDSY(KK,I)
                 ZBOLD(J,I) = CBDSZ(KK,I)
              ENDDO
           ENDDO
           !  Bisect, staging or random cubical MC move
           IF (BISE .or. STAGE) THEN
              IF (BISE) CALL BISECT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IBMOVE,NBMOVE,LAMBDA,KLEV)
              IF (STAGE) THEN
                 IF (OSTAGE) THEN
!!! NOTE: Mass-perturbation has to be worked out!!!
                    CALL OSTAGING(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,LAMBDA,NBEADSQQ,EED(NCON),ISTR,ISTRVEC)
                    EEDOLD(NCON+1) = EED(NCON)  ! Update previous end2end distance
                    IF (QFEP) EEDOLD2(NCON+1) = EED2(NCON)
                 ELSE
                    CALL STAGING(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IBMOVE,NBMV,LAMBDA,NBEADSQQ)
                 ENDIF
              ENDIF
              !  Enforce centroid constraint for all the particles in bisection
              CALL FPICNT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,NBEADSQQ)
           ELSE
              CALL MCMV(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IBMOVE,NBMOVE,LAMBDA,LAMBDA2,PIMCMV)
              !  Enforce centroid constraint for the moved particles only for metropolis
              CALL FPICNT2(CBDSX,CBDSY,CBDSZ,XBOLD,YBOLD,ZBOLD,NPIATM,NBEADSQQ,NBMV,IBMOVE)
           ENDIF
        ENDIF   ! MYNOD
        !
        ! Transfer bead coordinates and auxilliary MC array to all processors
        !
        if(numnod.gt.1) then
           CALL PSND8m(CBDSX,BUFSIZ)
           CALL PSND8m(CBDSY,BUFSIZ)
           CALL PSND8m(CBDSZ,BUFSIZ)
           CALL PSND4(IBMOVE,NBEADSQQ)
        endif
        !  Get energy of polymer rings
        EFPIHNEW = FPIHARM(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQQ,IPIATM,AMASS,FPIFAC,OSTAGE)
        ! Mass perturbation - no MC check for free-particles will be done since QFEP uses BISE
        IF (QFEP) THEN
           CALL MASSPERT(CBDSX,CBDSY,CBDSZ,PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,MSQRAT)
           EFPIHNEW2 = FPIHARM(PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQQ,IPIATM,WMAIN,FPIFAC,OSTAGE)
        ENDIF
        !
        !   Get dH only if not sampling accurate free particle distribution
        !   (e.g. bisecting).
        IF (BISE .or. STAGE) THEN
           DELH = ZERO
        ELSE
           DELH = EFPIHNEW-EFPIHOLD
        ENDIF
        !       if(prnlev.ge.2) write(6,*) 'delh',delh
        BFACT = EXP(-DELH*BETA_FPI)
        IF (MYNOD.EQ.0) XTMP = RANUMC()
        if(numnod.gt.1) CALL PSND8(XTMP,1)
        !       write(99,'(A,5F10.5)')'efpihnew,efpihold,delh,bfact,xtmp',efpihnew,efpihold,delh,bfact,xtmp
        !
        !   cDTM MC if
        !   Averaging over free beads
        IF(BFACT.GE.XTMP) THEN
           !
           !  Accept the old MC configuration, and update.
           !  New configuration is updated until the next acceptance move.
           !
           !  Don't do qm calculation during equilibration part
           IF (ITYP.EQ.2) THEN

              ETNE_FPI = ETOL_FPI
              IF (QFEP) ETNE_FPI2 = ETOL_FPI2
              DO J = ISTA,IEND
                 KK = IBMOVE(J)
                 DO I = 1,NPIATM
                    JJ = IPIATM(I)
                    X(JJ) = XOLD(I)+CBDSX(KK,I)
                    Y(JJ) = YOLD(I)+CBDSY(KK,I)
                    Z(JJ) = ZOLD(I)+CBDSZ(KK,I)
                 ENDDO
                 !  get energy
                 CALL PIENER(EFPIMOV(J),X,Y,Z,DX,DY,DZ,.FALSE.)
                 IF (TIAC .OR. CHAC) EFPIMOV(J)=EFACT(J)*EFPIMOV(J)+FFACT(J)*PITI(DX,DY,DZ)
                 IF (OSTAGE .AND. (J.EQ.1 .OR. J.EQ.NBEADSQQ)) EFPIMOV(J) = EFPIMOV(J)/TWO ! Average of first and last bead for open chain PI
                 ! cDTM Average over moved configs
                 ETNE_FPI = ETNE_FPI+(EFPIMOV(J)-EFPIOLD(KK))/(DBLE(NBEADSQ)*CHFACN)
       !           if(prnlev.ge.2) write(6,'(A,2I,3F15.5)') 'mc',ncon,j,efpimov(j),EFPIOLD(KK),etne_fpi
                 !
                 ! Mass perturbation
                 IF (QFEP) THEN
                    DO I = 1,NPIATM
                       JJ = IPIATM(I)
                       X(JJ) = XOLD(I)+PCBDSX(KK,I)
                       Y(JJ) = YOLD(I)+PCBDSY(KK,I)
                       Z(JJ) = ZOLD(I)+PCBDSZ(KK,I)
                    ENDDO
                    CALL PIENER(EFPIMOV2(J),X,Y,Z,DX,DY,DZ,.FALSE.)
                    IF (TIAC .OR. CHAC) EFPIMOV2(J)=EFACT(J)*EFPIMOV2(J)+FFACT(J)*PITI(DX,DY,DZ)
                    IF (OSTAGE .AND. (J.EQ.1 .OR. J.EQ.NBEADSQQ)) EFPIMOV2(J) = EFPIMOV2(J)/TWO ! Average of first and last bead for open chain
                    ETNE_FPI2 = ETNE_FPI2+(EFPIMOV2(J)-EFPIOLD2(KK))/(DBLE(NBEADSQ)*CHFACN)
                 ENDIF
              ENDDO   ! J = ISTA,IEND
              ETNE_FPI_ALL = ETNE_FPI
#if KEY_SQUANTM==1
              if (numnod.gt.1) CALL GCOMB(ETNE_FPI_ALL,1)
#endif 
              IF (QFEP) THEN
                 ETNE_FPI_ALL2 = ETNE_FPI2
#if KEY_SQUANTM==1
                 if (numnod.gt.1) CALL GCOMB(ETNE_FPI_ALL2,1)
#endif 
              ENDIF
              !   Sum up action
              IF (MYNOD.EQ.0) THEN
                 EBETA(NCON) = EXP(-BETA_FPI*(ETOL_FPI_ALL-ECL_XBAR))
                 PISUM = PISUM+XREPEAT*EBETA(NCON)
                 EPOTAVE = EPOTAVE+XREPEAT*ETOL_FPI_ALL
                 ! Mass perturbation
                 IF (QFEP) THEN
                    EBETA2(NCON) = EXP(-BETA_FPI*(ETOL_FPI_ALL2-ECL_XBAR))
                    PISUM2 = PISUM2+XREPEAT*EBETA2(NCON)
                    EPOTAVE2 = EPOTAVE2+XREPEAT*ETOL_FPI_ALL2
                 ENDIF
              ENDIF
              ETOL_FPI = ETNE_FPI
              ETOL_FPI_ALL = ETNE_FPI_ALL
              ! Mass perturbation
              IF (QFEP) THEN
                 ETOL_FPI2 = ETNE_FPI2
                 ETOL_FPI_ALL2 = ETNE_FPI_ALL2
              ENDIF
              DO J = ISTA,IEND
                 KK = IBMOVE(J)
                 EFPIOLD(KK) = EFPIMOV(J)
                 IF (QFEP) EFPIOLD2(KK) = EFPIMOV2(J)
              ENDDO
              ! Make sure energy arrays on different nodes are the same
#if KEY_SQUANTM==1
              if (numnod > 1) then
                 CALL VDGBRE(EFPIOLD,JPARPT)
                 if (QFEP) CALL VDGBRE(EFPIOLD2,JPARPT)
              endif
              ! End ITYP IF
#endif 
           ENDIF    ! ITYP.EQ.2

           IF (MYNOD.EQ.0) THEN
              XACCEPT = XACCEPT+XREPEAT
              EHARMAVE = EHARMAVE+XREPEAT*EFPIHOLD
              IF(QFEP) EHARMAVE2=EHARMAVE2+XREPEAT*EFPIHOLD2
              XREPEAT = ONE
           ENDIF
           EFPIHOLD = EFPIHNEW
           ! Mass perturbation
           IF (QFEP) EFPIHOLD2 = EFPIHNEW2

           !   cDTM MC else
        ELSE    ! BFACT.GE.XTMP
           !
           ! Bring back the original coordinates
           ! Only if Metropolis - bisect and staging have 100% acceptance
           IF (MYNOD.EQ.0) THEN
              DO I = 1,NPIATM
                 DO J = 1,NBEADMV
                    KK = IBMOVE(J)
                    CBDSX(KK,I) = XBOLD(J,I)
                    CBDSY(KK,I) = YBOLD(J,I)
                    CBDSZ(KK,I) = ZBOLD(J,I)
                 ENDDO
              ENDDO
              XREJECT = XREJECT+ONE
              XREPEAT = XREPEAT+ONE
           ENDIF   ! MYNOD
           if(numnod.gt.1) then
              CALL PSND8m(CBDSX,BUFSIZ)
              CALL PSND8m(CBDSY,BUFSIZ)
              CALL PSND8m(CBDSZ,BUFSIZ)
           endif
           !   cDTM MC endif
        ENDIF  ! BFACT.GE.XTMP

        ! Print potential and kinetic energy at each FP iteration
        IF (MYNOD.EQ.0 .AND. PRNLEV.GT.4 .AND. ITYP.EQ.2) THEN
           IF (QFEP) THEN
              WRITE (OUTU,'(A,I7,F20.5,F20.5,F12.5)') ' QPIMC> ',NCON,ETNE_FPI_ALL,ETNE_FPI_ALL2,EFPIHNEW
           ELSE
              WRITE (OUTU,'(A,I7,F20.5,F12.5)') ' QPIMC> ',NCON,ETNE_FPI_ALL,EFPIHNEW
           ENDIF
        ENDIF
        ! cDTM end do NCONFS
     ENDDO loopNCON

     IF (MYNOD.EQ.0) THEN
        if (prnlev.ge.2) then
           WRITE(OUTU,'(A,I12)') ' QPIMC> NCONF: ',NCON-1
           WRITE(OUTU,'(A,I12)') ' QPIMC> NRJCT: ',INT(XREJECT)
           WRITE(OUTU,'(A,I12)') ' QPIMC> NACCP: ',INT(XACCEPT)
           IF(XACCEPT.GT.ZERO)WRITE(OUTU,'(A,F8.4)')' QPIMC> ACC %: ',100.0D0*(NCON-XREJECT-1)/(NCON-1)
        endif
        !
        !  averaging
        !
        IF(XACCEPT.EQ.ZERO) XACCEPT = DBLE(NCONFS)
        !      if(prnlev.ge.2) write(6,*) 'accept:',int(xaccept)
        !
        EHARMAVE = EHARMAVE/XACCEPT
        ! Mass perturbation
        IF (QFEP) EHARMAVE2 = EHARMAVE2/XACCEPT
     ENDIF
     !   cDTM end do ITYP
  ENDDO loopITYP


  !
  !   Restore original coordinates
  DO I = 1,NPIATM
     JJ = IPIATM(I)
     X(JJ) = XOLD(I)
     Y(JJ) = YOLD(I)
     Z(JJ) = ZOLD(I)
  ENDDO
  IF (MYNOD.EQ.0) THEN
     ! Write open chain PI simulation data - weighting factor and end2end bead
     ! distance
     IF (OSTAGE) THEN
        RXNCOR = GETRXN(X,Y,Z,RXNA,RXNB,RXNC)
        WRITE(NMOMDIS,'(I7,F18.5)') NCONFS,RXNCOR
        IF (QFEP) WRITE(NMOMDIS2,'(I7,F18.5)') NCONFS,RXNCOR
        DO I = 1,NCONFS
           WRITE(NMOMDIS,'(E18.8,F12.5)') EBETA(I),EEDOLD(I)
           IF (QFEP) WRITE(NMOMDIS2,'(E18.8,F12.5)') EBETA2(I),EEDOLD2(I)
        ENDDO
     ENDIF
     PIAVE = PISUM/XACCEPT
     EPOTAVE = EPOTAVE/XACCEPT
     !              WRITE(NQUBOUT,'(A,E18.8)') ' QPIMC> PIAVE = ',PIAVE
     if(prnlev.ge.2) WRITE(OUTU,'(A,E18.8)') ' QPIMC> PIAVE = ',PIAVE
     ! Mass perturbation
     IF (QFEP) THEN
        PIAVE2 = PISUM2/XACCEPT
        EPOTAVE2 = EPOTAVE2/XACCEPT
        if(prnlev.ge.2) WRITE(OUTU,'(A,E18.8)') ' QPIMC> PIAVE = ',PIAVE2
     ENDIF
     !
     !  write out all beads coordinates
     !
     IF(NBOUT_UNIT.GT.0) THEN
        WRITE (NBOUT_UNIT,*) NPIATM,NBEADSQQ
        DO I = 1,NPIATM
           WRITE (NBOUT_UNIT,'(6F12.8)') (CBDSX(J,I),CBDSY(J,I),CBDSZ(J,I),J=1,NBEADSQQ)
        ENDDO
     ELSE
        WRITE(OUTU,'(A)') ' QPIMC> Final bead coordinates not written'
     ENDIF
  ENDIF
  if(numnod.gt.1) CALL PSND8(PIAVE,1)
  !
  !
  call chmdealloc('qub.src','QMPIMC','EBETA',EESIZE,crl=EBETA)
  call chmdealloc('qub.src','QMPIMC','EBETA2',EESIZE,crl=EBETA2)
  call chmdealloc('qub.src','QMPIMC','EED',EESIZE,crl=EED)
  call chmdealloc('qub.src','QMPIMC','EED2',EESIZE,crl=EED2)
  call chmdealloc('qub.src','QMPIMC','EEDOLD',EESIZE,crl=EEDOLD)
  call chmdealloc('qub.src','QMPIMC','EEDOLD2',EESIZE,crl=EEDOLD2)
  !
  RETURN
END SUBROUTINE QMPIMC

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Divide a loop over processors
! From IBM MPI Programming publication
! N1 & N2 - beginning & end of non-parallel loop
! NPROCS - # of processors
! IRANK - processor rank
! ISTA & IEND - returned beginning & end of parallel loop for processor
!               with rank IRANK
SUBROUTINE PARA_RANGE(N1,N2,NPROCS,IRANK,ISTA,IEND)

  implicit none

  INTEGER N1,N2,NPROCS,IRANK,ISTA,IEND,IWORK1,IWORK2

  IWORK1 = (N2-N1+1)/NPROCS
  IWORK2 = MOD(N2-N1+1,NPROCS)
  ISTA = IRANK*IWORK1 + N1 + MIN(IRANK,IWORK2)
  IEND = ISTA + IWORK1 - 1
  IF (IWORK2.GT.IRANK) IEND = IEND + 1

  RETURN

END SUBROUTINE PARA_RANGE
#endif /* (pll)*/
#endif /* (quant)*/

end module qub_m

