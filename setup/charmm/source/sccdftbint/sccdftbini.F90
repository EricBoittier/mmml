#if KEY_SCCDFTB==0
SUBROUTINE SCCTBINI(COMLYN,COMLEN) 
  CHARACTER COMLYN*(*)
  INTEGER   COMLEN
  CALL WRNDIE(-1,'<SCCTBINI>','SCCDFTB code not compiled.')
return
end SUBROUTINE SCCTBINI

#else /* KEY_SCCDFTB */

SUBROUTINE SCCTBINI(COMLYN,COMLEN) 
  !-----------------------------------------------------------------------
  !     This is the interface for running SCCDFTB with CHARMM for
  !     QM/MM calculations. It closely follows the interface 
  !     developed for GAMESS/CADPAC/DeFT.
  !
  !     Q. Cui, M. Elstner Feb. 1999.
  !     Most recently updated, June. 2002 
  !     
  !     G. Li, Q. Cui, 2003. 
  !     Added DTSC, pKa free energy simulations
  !
  !     Patti Schaefer, Q. Cui, Mar. 2004
  !
  !     Added eWald summation and GSBP
  !
  !     Several aspects remain to be fixed/improved in the near future:
  !     4. Dispersion interactions
  !     5. Better compatability with path-integral and Tsallis statistics
  !
  !     Functions to be added related to SCCDFTB:
  !     1. Open-shell treatment
  !     2. Time-dependent treatment
  !
  use dimens_fcm
  use string
  use blockscc_fcm
  use code
  use coord
  use consta
  use number
  use param
  use psf
  use select
  use stream
  use energym
  use replica_mod
  use sccdftb
  use sccdftbsrc 
  use gamess_fcm
  use parallel     
  use memory
  use image
  !     Enforce FAST OFF in free energy calculations
  use fast
  use stbgho
  use gukini_mod,only:recopsel
  ! PZ MG_QC_UW1206: pmeutil
  use sccpmeutil, only: nfft1,nfft2,nfft3,forder
  use sccpme_module, only: PMESH_SETUP
  ! PZ: pmeutil end

  implicit none

  integer,allocatable,dimension(:) :: immlst1
  real(chm_real),allocatable,dimension(:,:) :: cptc1
  real(chm_real),allocatable,dimension(:) :: zptc1
  ! QC: 11/17 for PERT based on Xiya
  ! integer,allocatable,dimension(:) :: igmsel1    ! moved to blockscc_fcm
  real(chm_real),allocatable,dimension(:) :: cgtmp ! Xiya
  ! QC: 11/17 done
  integer,allocatable,dimension(:) :: ISLCT, ISLCT1,ISLCT2
  integer,allocatable,dimension(:) :: LSLCT
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  LOGICAL   QQINP

  INTEGER II,jj
  !
#if KEY_REPLICA==1 /*replica*/
#if KEY_PARALLEL==1
  INTEGER IDSTNC(MAXNODE)         
#endif
  INTEGER I,J,IPT,IREPL,KK,IBLQM,IBL
#endif /* (replica)*/
  !
  !

  !     --------------- First process multi-topology related stuff ----
  qlamda=(INDXA(COMLYN,COMLEN,'LAMD').GT.0)
  qpkac=(INDXA(COMLYN,COMLEN,'PKAC').GT.0)
  qstop=(INDXA(COMLYN,COMLEN,'STOP').GT.0)
  qdtop=(INDXA(COMLYN,COMLEN,'DTOP').GT.0)
  ! QC: 11/17 Add PERT
  qsccpert=(INDXA(COMLYN,COMLEN,'PERT').GT.0)
  ! QC: 11/17 Done

  !     QC_UW04: enforce FAST OFF if free energy calc is selected
  if(qlamda.or.qpkac.or.qstop.or.qdtop) then
     faster=-1
     lfast=-1
  endif

  if(qlamda.or.qpkac) then
     outpka=GTRMI(COMLYN,COMLEN,'OUTP',15)
  endif

  if(qpkac) then
     idxstp=GTRMI(COMLYN,COMLEN,'ISTP',1)
     idxhyd=GTRMI(COMLYN,COMLEN,'HYGN',1)
     cg(idxhyd)=0.0d0
     if(idxstp.eq.1) then
        qstop=.true.
        qlamda=.true.
        qdtop=.false.
        qsccb=.true.
     else
        qsccb=.true.
        sccpass=GTRMI(COMLYN,COMLEN,'PASS',10000)
        sccstep=GTRMI(COMLYN,COMLEN,'STEP',20)
        tico=GTRMF(COMLYN,COMLEN,'TICO', 1.0D-01)
        tiav=GTRMI(COMLYN,COMLEN,'TIAV',5)
        qsccres=(INDXA(COMLYN,COMLEN,'REST').GT.0)
        qlamda=.false.
     endif
  endif

  if(qstop.and.qlamda) then
     qstop=.true.
     qdtop=.false.
     qsccb=.true.
  else if(qlamda) then
     qdtop=.true.
     qstop=.false.
     qsccb=.true.
  endif

  !     ------------initial guess option------------------
  !     Guishan Zheng 03/17/2013
  iguess = GTRMI(COMLYN,COMLEN,'IGUE',0)
  idamp = GTRMI(COMLYN,COMLEN,'IDAM',100)
  if(iguess.gt.0) then
      IMD = 0
      qaux  = Zero          ! two dimension matrix; f90 compiler required
      Write(*,*) 'SCCTBENE: New Initial Guess algorithm is selected'
  endif
  !     ------------------------ SCC selections done here -------------

  qmused = .true.      ! safe here since next are the allocations, ...
  call allocate_gamess ! try to reduce from MAXA to NATOM

  ! JZ_UW12
  SMBP_GRAD  = .false.
  SCC_CALLED = .false.

  call chmalloc('sccdftbini.src','SCCTBINI','ISLCT',NATOM,intg=ISLCT)
  call SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !
  ! conditionally select GHO boundary atoms if there are any
  !                                                 ... PJ 7/2004
  QLINK = (INDXA(COMLYN, COMLEN, 'GLNK') .GT. 0)
  IF (QLINK) THEN
     IF (PRNLEV .GE. 2) THEN
        WRITE(OUTU,22) ('GLNK: GHO boundary atoms are used')
     END IF
     call chmalloc('sccdftbini.src','SCCTBINI','LSLCT',NATOM,intg=LSLCT)
     call SELCTA(COMLYN,COMLEN,LSLCT,X,Y,Z,WMAIN,.TRUE.)
     call GHOHYB(ISLCT,LSLCT)
  ENDIF
  !
  QGMREM=(INDXA(COMLYN,COMLEN,'REMO').GT.0)

#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0)THEN
#endif 
     IF(QGMREM) THEN
        WRITE(OUTU,22) &
             'REMOve: Classical energies within QM atoms are removed.'
     ELSE
        WRITE(OUTU,22) &
             'No REMOve: Classical energies within QM atoms are retained.'
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
  QGMEXG=(INDXA(COMLYN,COMLEN,'EXGR').GT.0)
  !     XIAO_QC_UW0609: DIV Scheme
  QSCCDIV = (INDXA(COMLYN,COMLEN,'DIV').GT.0)
  IF (QSCCDIV) then
     QGMEXG  = .FALSE.
  ENDIF
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0)THEN
#endif 
     IF(QGMEXG) THEN
        WRITE(OUTU,22) &
             'EXGRoup: QM/MM Electrostatics for link host groups removed.'
        !     XIAO_QC_UW0609 DIV Scheme
     ELSE IF(QSCCDIV) THEN
        WRITE(OUTU,22) &
             'Using DIV scheme: using standard SLA scheme for exclusions.'
     ELSE
        WRITE(OUTU,22) &
             'No EXGRoup: QM/MM Elec. for link atom host only is removed.'
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
  QQINP=(INDXA(COMLYN,COMLEN,'QINP').GT.0)
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0)THEN
#endif 
     IF(QQINP) THEN
        WRITE(OUTU,22) &
             'QINP: Charges will input for QM atoms.'
     ELSE
        WRITE(OUTU,22) &
             'No QINP: Charges will be based on atomic numbers.'
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
22 FORMAT(' SCCINT> ',A)
23 FORMAT(' SCCINT> ',A,F12.8) ! MG_QC_UW1207 lcolspin
  !
  !     Additional control parameters for SCCDFTB 
  !     TELEC: electronic temperature
  !     SCFTOL, MXITSCF: self-explanatory
  !     LPRVEC: whether to print eigenvectors
  !     ICHSCC: charge of the system

  TELEC=GTRMF(COMLYN,COMLEN,'TEMP', ZERO) 
  SCFTOL=GTRMF(COMLYN,COMLEN,'SCFT', 1.0D-07) 
  MXITSCF=GTRMI(COMLYN,COMLEN,'MXIT',50)
  LPRVEC=(INDXA(COMLYN,COMLEN,'PRVC').GT.0)  
  ICHSCC=GTRMI(COMLYN,COMLEN,'CHRG',0)

! Guanhua_puja_QC_UW1212: use which mixer
! 0: simple mixing
! 1: Anderson mixing
! 2: Broyden mixing
  SCCMIX=GTRMI(COMLYN,COMLEN,'MIXE',2)
! Guanhua_puja_QC_UW1212: how many generations for Anderson mixing
  GENS=GTRMI(COMLYN,COMLEN,'GENS',8)
! Guanhua_puja_QC_UW1212: symmetry breaking number for Anderson mixing
  WSY=GTRMF(COMLYN,COMLEN,'SYMM',0.01d0)
!
#if KEY_DFTBMKL==0
  IF ((SCCMIX.EQ.1).OR.(SCCMIX.EQ.3)) THEN
    SCCMIX=2
    write(outu,22)'Linking to MKL not available. Using Broyden mixer &
   &  instead of Anderson/DIIS'
  ENDIF
#endif 
!
  ! JZ_UW12: Add options for saving/reading 'hessian' (= DD1) for 
  !          finite-difference frequency calculations
  QSCCREADHESS=(INDXA(COMLYN,COMLEN,'RHES').GT.0)
  QSCCSAVEHESS=(INDXA(COMLYN,COMLEN,'SHES').GT.0)

  ! MG_QC_UW1206: spin
  lcolspin=.false.
  unpe=GTRMF(COMLYN,COMLEN,'UNPE',ZERO) 
  if (unpe.gt.1.0d-12) then
    write(outu,22) 'Collinear spin-formalism will be applied (JPCA2007,111,5622)'
    write(outu,23) 'number of unpaired electrons: ',unpe
    lcolspin=.true.
    ! no compatibility with GLNK 
    if (qlink) call WRNDIE(-5,'<SCCDFTBINI>','UNPE is not currently not compatible with GLNK') 
  endif

  ! MG+Guanhua_QC_UW1206: KO
  lcdko=(INDXA(COMLYN,COMLEN,'CDKO') .gt. 0)
  if (lcdko) then
    write(outu,22) "Use charge-dependent Klopman-Ohno formalism for QM/MM interactions"
    call chardkoread
  endif


  ! ASC: Use DFTB+ or not?
  ldftbplus = .false.
  ldftbplus=(INDXA(COMLYN,COMLEN,'PLUS').GT.0)
  if (ldftbplus) then
#if KEY_DFTBPLUS==1
      write (outu,22) "Running SCC-DFTB via DFTB+ interface."
#else
      write (outu,22) "CHARMM not compiled with +DFTBPLUS keyword."
      write (outu,22) "Running SCC-DFTB via EGLCAO interface instead."
      ldftbplus = .false.
#endif
  else
      write (outu,22) "Running SCC-DFTB via EGLCAO code."
  endif

  ! XL_QC_UW1211: Diagonalizer
  DMETH=GTRMI(COMLYN,COMLEN,'DMET',2)

#if KEY_DFTBMKL==0
  if (DMETH .ne. 0) then
    write (outu,22) "LAPACK DSYGV* requested as diagonalizer, but DFTBMKL is not present."
    DMETH = 0
  endif
#endif

  if (dmeth .eq. 0) then
    write (outu,22) "Use EWEVGE as diagonalizer"
  elseif (dmeth .eq. 1) then
    write (outu,22) "Use LAPACK DSYGV as diagonalizer"
  elseif (dmeth .eq. 2) then
    write (outu,22) "Use LAPACK DSYGVD as diagonalizer"
  else
    write (outu,22) "Wrong diagonalizer option"
    stop
  endif

! STRT QC_UW_1017, based on XL_QC_UW1505 NBO
  QNBO=(INDXA(COMLYN,COMLEN,'NBO').GT.0)
  IF (QNBO) WRITE(outu,*) "NBO analysis requested"

! END  QC_UW_1017, based on XL_QC_UW1505 NBO

  !     -------------- for lamda from guohui ---------------------
  if(qlamda) then
     scclamd=GTRMF(COMLYN,COMLEN,'INIT', 1.0D-02)
     ! sccpass:  how many initial (MDSimu) steps to be excluded 
     ! in computing <dvdl>
     sccpass=GTRMI(COMLYN,COMLEN,'PASS',10000)
     ! sccstep:  skip every scccstep to compute one value for dvdl.
     sccstep=GTRMI(COMLYN,COMLEN,'STEP',20)
     ! tico   : TI-convergence 
     tico=GTRMF(COMLYN,COMLEN,'TICO', 1.0D-01)
     ! tiav   : number of steps you do one time
     ! average for evaluting <dvdl>
     tiav=GTRMI(COMLYN,COMLEN,'TIAV',5)

     ! qsccres is just for restart calculations for SCC-DFTB-TI
     qsccres=(INDXA(COMLYN,COMLEN,'REST').GT.0)
     !         if(qsccres) then
     ! icntdyn1 is just for restart calculations for SCC-DFTB-TI
     !           icntdyn1=GTRMI(COMLYN,COMLEN,'COUN',1000)
     ! iavti1 is just for restart calculations for SCC-DFTB-TI
     !           iavti1=GTRMI(COMLYN,COMLEN,'NAVE',1000)
     ! dvdl2 is just for restart calculations for SCC-DFTB-TI
     !           dvdl2=GTRMF(COMLYN,COMLEN,'DVDL', 1.0D-02)
     ! dvdlav1 is just for restart calculations for SCC-DFTB-TI
     !           dvdlav1=GTRMF(COMLYN,COMLEN,'ADVD', 1.0D-02)
     !           if(sccnstr.gt.0) then
     !           aecnstr1=GTRMF(COMLYN,COMLEN,'AEXP', 1.0D-02)
     !           endif
     !         endif

     TELEC1=telec
     SCFTOL1=scftol
     MXITSCF1=mxitscf
     LPRVEC1=lprvec
     ICHSCC1=ichscc
     !MG_121101: extension to lcolspin
     UNPE1=unpe 
     LCOLSPIN1=lcolspin
     !     ------------- Second state --------------------

     TELEC2=GTRMF(COMLYN,COMLEN,'TEMP', 1.0D-02)
     SCFTOL2=GTRMF(COMLYN,COMLEN,'SCFT', 1.0D-06)
     MXITSCF2=GTRMI(COMLYN,COMLEN,'MXIT',50)
     LPRVEC2=(INDXA(COMLYN,COMLEN,'PRVC').GT.0)
     ICHSCC2=GTRMI(COMLYN,COMLEN,'CHRG',0)
     UNPE2=GTRMF(COMLYN,COMLEN,'UNPE',ZERO) 
     if (UNPE2.gt.1.0d-12) then
       write(outu,23) 'number of unpaired electrons for second state: ',UNPE2
       LCOLSPIN2=.true.
     elseif (lcolspin1) then
       write(outu,22) 'No Collinear spin-formalism applied for the second state'
     endif
  endif

  !     QC: 11/17 Add PERT 
!     -------------- for sccpert Xiya ---------------------
  if(qsccpert) then 

     ! first state
     TELEC1=telec
     SCFTOL1=scftol
     ICHSCC1=ichscc
     !MG_121101: extension to lcolspin
     UNPE1=unpe
     LCOLSPIN1=lcolspin

     ! second state
     TELEC2=GTRMF(COMLYN,COMLEN,'TEMP', 1.0D-02)
     SCFTOL2=GTRMF(COMLYN,COMLEN,'SCFT', 1.0D-06)
     ICHSCC2=GTRMI(COMLYN,COMLEN,'CHRG',0)
     UNPE2=GTRMF(COMLYN,COMLEN,'UNPE',ZERO)
     if (UNPE2.gt.1.0d-12) then 
       write(outu,23) 'number of unpaired electrons for second state: ',UNPE2
       LCOLSPIN2=.true.
     elseif (lcolspin1) then 
       write(outu,22) 'No Collinear spin-formalism applied for the second state'
     endif
  endif

  !     QC: 11/17 End
  !     ----------------------------------------------------------------
  !     QC_UW031205: Add dispersion option 

  dispers=(INDXA(COMLYN,COMLEN,'DISP').GT.0)
  if (dispers) WRITE(OUTU,*) "Dispersion among QM atoms included"

  !     ----------------------------------------------------------------
  !     QC: WILL COPY RCUTQM FROM the MKMMLST CODE

  qsccnb=(INDXA(COMLYN,COMLEN,'CUTF').GT.0)
  if (qsccnb) then
     WRITE(OUTU,'(1x,"SCCINT> USE CUT-OFF FOR QM/MM")')
  else
     WRITE(OUTU,'(1x,"SCCINT> DO NOT USE CUT-OFF FOR QM/MM")')
  endif
  !     -----------------------------------------------------
  !     QC_UW04: Add ewald / Mulliken
  !     we have the option of letting SCC optimize ewald summation
  !     parameters (for QM/QM and QM/MM), or take the same from CHARMM

  !     IF pack image atoms for the QM region: for most cases, YES
  !     for protein simulations with the QM region localized very 
  !     far from the boundary - avoid packing image can save quite some
  !     time

  LSKIMG=(INDXA(COMLYN,COMLEN,'SKIM').GT.0)
  period =(INDXA(COMLYN,COMLEN,'EWAD').GT.0) 

  IF (period) THEN
     iuptbox=GTRMI(COMLYN,COMLEN,'UPDT',0)
     !       QC_UW031205: 
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0)THEN
#endif 
        write(outu,*) "SCCDFTBINI> eWald is requested"
        write(outu,*)  &
             "SCCDFTBINI> Note that self-interaction correction not included"
        write(outu,*)  &
             "SCCDFTBINI> For cube:0.5*(-2.837297/L)*Sum{qi*qi}"
        write(outu,*) "SCCDFTBINI> See, Hummer et al, JPC, 100, 1206"
        write(outu,*) "SCCDFTBINI> Volume correction term included."
        write(outu,*) "SCCDFTBINI> See, Brooks et al, JCP, 108, 7070"
        write(outu,*) "SCCDFTBINI> UPDATE LATTICE for QM ",iuptbox
        !      QC_UW031205
#if KEY_PARALLEL==1
     ENDIF
#endif 

     !       if choose parameters for QM/MM ewald
     LSCOEWD=(INDXA(COMLYN,COMLEN,'EOPT').GT.0) 
     IF (LSCOEWD) THEN
        !      QC_UW031205
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0)THEN
#endif 
           write(outu,*) "Perform self-consistent opt for QM eWald"
           !      QC_UW031205
#if KEY_PARALLEL==1
        ENDIF
#endif 
     ELSE
        !       here we specify eWald parameters explicitly - to be consistent
        !       with CHARMM. We also will do the real-space part in the same
        !       way as CHARMM - i.e., instead of testing convergence with 
        !       respect to the summation, we truncate based on cutoff
        !       schemes.
        kappascc=GTRMF(COMLYN,COMLEN,'KAPP', 0.34d0)
        kmxscc=GTRMI(COMLYN,COMLEN,'KMAX',10)
        ksqmxscc=GTRMI(COMLYN,COMLEN,'KSQM',200)
        !       convert the unit of KAPP ! BUGFIX MG_QC_UW1205
        kappascc=kappascc*BOHRR
        !      QC_UW031205
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0)THEN
#endif 
           write(outu,*) "Specify eWald parameters:"
           write(outu,*) "KAPPA: ",kappascc," KMAX: ",kmxscc, &
                "KSQMAX: ",ksqmxscc
           !      QC_UW031205
#if KEY_PARALLEL==1
        ENDIF
#endif 
     ENDIF
     call updbox(boxsiz,XTLABC,0,NLAT)
  ENDIF

  !     SCC_ewald end
  !     PZ: SCC_PMEwald begin   MG_QC_UW1206
      qsccpme=(INDXA(COMLYN,COMLEN,'PMEW').GT.0)
      fftx=GTRMI(COMLYN,COMLEN,'FFTX',32)
      ffty=GTRMI(COMLYN,COMLEN,'FFTY',32)
      fftz=GTRMI(COMLYN,COMLEN,'FFTZ',32)
      fftorder=GTRMI(COMLYN,COMLEN,'ORDE',6)
      nfft1=fftx
      nfft2=ffty
      nfft3=fftz
      forder=fftorder
      if (qsccpme) then
        call pmesh_setup(NATOM,xnsymm)
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0)THEN
#endif 
          write(outu,*) "PMEWald parameters for QM/MM"
          write(outu,*) "order: ",forder," fftx: ",fftx," ffty: ",ffty," fftz: ",fftz
          write(outu,*) "xnsymm",xnsymm
#if KEY_PARALLEL==1
        ENDIF
#endif 
      endif
  !     PZ: SCC_PMEwald end

  !     QC: Newly added - if mulliken population is to be added
  !     we would like to transfer it into the CG array such that it 
  !     can be displayed. We assign QMULIK to CG after all energy calls
  !     and clear QM's CG before each SCC call

  LMULIK=(INDXA(COMLYN,COMLEN,'MULL').GT.0)
  IF (LMULIK) WRITE(outu,*) "Mulliken population requested"

  !     Haibo_QC_UW0609 Mulliken convergence and Dipole Output
  LSCFCHG=(INDXA(COMLYN,COMLEN,'SCFC').GT.0)
  if (LSCFCHG) write(outu,'(a)') 'SCCINT> Tight SCF based on Mulliken'
  ! MG_QC_UW1207
  ! lscfchg in combination with lcolspin is not very nicely defined as it just compares the total orbital-resolved
  ! mulliken charge but not spinup and spindown separately..., however this criteria is additionally to the energy
  ! tolerance criteria!
  !if ((LSCFCHG).AND.(lcolspin)) call wrndie(-5,'<SCCDFTBINI>','UNPE is not currently not compatible with SCFC')
  LSCCDIP=(INDXA(COMLYN,COMLEN,'QDIP').GT.0)
  if (LSCCDIP) then 
     if (.not.LMULIK) then 
        LMULIK=.TRUE.
        write(outu,'(a)') ' SCCINT> Mulliken turned on'
     endif
     UDIP  = GTRMI (COMLYN,COMLEN,'UDIP',-1)
     if(UDIP.gt.0)then
        write(outu,'(a,i3)') &
             ' SCCINT> Mulliken/Coor will be written to unit ', UDIP 
     else 
        call wrndie(-5,'<SCCTBINI>','please provide a positive unit for UDIP')
     endif
  endif

  !     QC: Whether to read in Hubbard derivative parameters from the end
  !     of the sccdftb file 

  luhder=(INDXA(COMLYN,COMLEN,'DHUB').GT.0)
  IF (LUHDER) THEN 
     WRITE(OUTU,*) "USE CHARGE-DEPEDENT HUBBARD"
     WRITE(OUTU,*) "WARNING: ASSUMING L-INDEPENDENT"
  ENDIF

  !     QC UW_06: Add Gaussian adjustment
  luhgau=(INDXA(COMLYN,COMLEN,'DHGA').GT.0)
  IF (LUHGAU) THEN 
     WRITE(OUTU,*) "INCLUDE GAUSSIAN IN HUBBARD DERIVATIVE"
     LUHDER=.TRUE.
  ENDIF

  !     QC: Another Gaussian fix as SRP to the repulsive potential
  LGAUSW=(INDXA(COMLYN,COMLEN,'GAUS').GT.0)
  IF (LGAUSW)  &
       WRITE(OUTU,*) "Use Additional Switchable Gaussian Repulsion"

  !     QC: Add hbond correction 
  LSCCHB=(INDXA(COMLYN,COMLEN,'HBON').GT.0)
  IF (LSCCHB) &
       WRITE(OUTU,*) "Use Hbond Correction"

  ! ASC: Grimmes DFT-D3 two body dispersion
  LSCCDFTD2=(INDXA(COMLYN,COMLEN,'TWOBOD').GT.0)

  ! ASC: Grimmes DFT-D3 two body dispersion
  LSCCDFTD3=(INDXA(COMLYN,COMLEN,'THREEBOD').GT.0)

  IF (LSCCDFTD3) THEN
       WRITE(OUTU,*) "DFTB: Using THREE-body DFT-D3(BJ)+E_abc dispersion correction."
  ELSE IF (LSCCDFTD2) THEN
       WRITE(OUTU,*) "DFTB: Using TWO-body DFT-D3(BJ) dispersion correction."
  ENDIF

  lcpeq=(INDXA(COMLYN,COMLEN,'CPEQ').GT.0)
  lcpe0=(INDXA(COMLYN,COMLEN,'CPE0').GT.0)

  if ((lcpeq).or.(lcpe0)) then
      lcpe = .true.
#if KEY_DFTBMKL==0
    
       call WRNDIE(-5,'<SCCTBINI>','CPE NOT POSSIBLE WITHOUT LINKING TO LAPACK (I.E. +DFTBMKL OPTION)')

#endif
  endif

  if (lcpeq) then
    write(*,*) "DFTB-D3/CPE - charge dependent - enabled"
  endif

  if (lcpe0) then
    write(*,*) "DFTB-D3/CPE - charge dependent - enabled"
  endif

  ! MG+Guanhua_QC_UW1101: Add full 3rd order
  lscc3rd=(INDXA(COMLYN,COMLEN,'D3RD').GT.0)
  IF (lscc3rd) WRITE(OUTU,*) "Use Full Third Order SCC"

  if ((lcpe.eqv..true.).and.(lscc3rd.eqv..false.)) then
       call WRNDIE(-5,'<SCCTBINI>','ERROR CPE ONLY IMPLEMENTED FOR DFTB3')      
  endif


  ! MG_UW1210: Add l-dependence for gamma-function/Hubbards (ldep)
  lldep=(INDXA(COMLYN,COMLEN,'LDEP').GT.0)
  IF (lldep) WRITE(OUTU,*) "Use l-dependence for Hubbard Parameters (ldep)"
  if (lldep.and.(luhder.or.luhgau)) &
       call WRNDIE(-5,'<SCCTBINI>','LDEP not implemented in connection with "DHUB" or "DHGA".')      

  !     QC_UW04: Done
  !     -----------------------------------------------------

  !     QC: SCC Relica, for convenience, perform replica scaling
  !     inside SCC
  NSCCRP=GTRMI(COMLYN,COMLEN,'NRPL',1)
  IF (NSCCRP.NE.1) LSCCRP=.TRUE.
  !
  !     QC: Removed LCNSCD (Constraints on centroid for pimd)

  MUSTUP=.TRUE.

  !PATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINT
  !PATHINT QC: WE NEED TO DO THE FOLLOWING TO USE SCC WITH PATHINT
  !PATHINT
  !PATHINT 1. Always do the replica first in the CHARMM script
  !PATHINT
  !PATHINT 2. Get # of the Replica and # of Atoms without replica,note
  !PATHINT    that not all QM atoms need to be replicated in general
  !PATHINT    but always has to do N_replica QM/MM calculations.
  !PATHINT    But why not then --- doesn't cost anything  in addition.
  !PATHINT 3. The tricky part is really the assignment part and the
  !PATHINT    BLOCK etc. business should be taken care of automatically.
  !PATHINT    Yet check bonded terms carefully (should be taken care of
  !PATHINT    provided that IGMSEL is assigned after replica.
  !PATHINT
  !PATHINT 4. Do N_replica QM/MM and scale accordingly.
  !PATHINT
  !PATHINT    For convenice, we take the # of replica and # of SCCAT
  !PATHINT    from CHARMM input, so CHECK CAREFULLY!
  !PATHINT 5. At the moment we can only do anti-symmetric streching
  !PATHINT    NUMC = total number of distance restraining

  !PATINT     Read in harmonic constraints
  LCNSCD=(INDXA(COMLYN,COMLEN,'UMCD').GT.0)
  IF (LCNSCD) THEN
     NCNSCD =GTRMI(COMLYN,COMLEN,'NUMC', 1)
     FCNSCD =GTRMF(COMLYN,COMLEN,'FUMC', 100.d0)
     UNTSCD =GTRMI(COMLYN,COMLEN,'RUN', 0)
     RCNSCD0 = GTRMF(COMLYN,COMLEN,'RREF0', ZERO)
     DO II=1,NCNSCD
        IATCNS((II-1)*3+1)= GTRMI(COMLYN,COMLEN,'AUM1',1)
        IATCNS((II-1)*3+2)= GTRMI(COMLYN,COMLEN,'AUM2',2)
        IATCNS((II-1)*3+3)= GTRMI(COMLYN,COMLEN,'AUM3',3)
     ENDDO

     !PATHINT      QC: NFRQSC1: how often do we record constraint dis
     !PATHINT      QC: NFRQSCW: how often do we write ths stuff to the disk
     NFRQSC1=GTRMI(COMLYN,COMLEN,'FRQR',1)
     NFRQSCW=GTRMI(COMLYN,COMLEN,'FRQW',1000)

#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0)THEN
#endif 
        WRITE(OUTU,*) "Harmonic constraint will be applied"
        WRITE(OUTU,'(1x," FORCE: ",F10.5," CENTER: ",F10.5)') &
             FCNSCD, RCNSCD0
        IF (UNTSCD.NE.0) THEN
           WRITE(OUTU,*) "FILE OPENED FOR CONS> ",UNTSCD
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif 
  ENDIF

  !PATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINT

#if KEY_REPLICA==1 /*repl*/
#if KEY_RPATH==1
  ! defaults for QM replica loop structure
  NPGRP      = 1                 ! number of parallel groups
  NREPQMNOD  = 1                 ! number of replicas on this node

#if KEY_PARALLEL==1 /*parallel*/
  !     save global node values
  NUMNODG=NUMNOD
  MYNODG=MYNOD
  QQMPAR=.FALSE.
  IF(NUMNOD.GT.1)QQMPAR=.TRUE.
  !
#endif /* (parallel)*/
#endif 
#endif /* (repl)*/
  !
#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1
  IF(QREP) THEN

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
              if(qmused_qchem) then
                 WRITE(OUTU,20)'QCHEM>', KREPQM
              else
                 WRITE(OUTU,20)'GUKINI>', KREPQM
              endif
20            FORMAT(a,' Parallel replicas activated, use ', &
                   i3,' procs per replica')
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
21            FORMAT('GUKINT> All replicas will run parallel ', &
                   'over all ',i3,' processors')
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


!  QC: 11/17 for PERT based on Xiya
!  call SCCTBSEL(ISLCT,QQINP)
  if(qsccpert) then 
      !keep a copy of original cg
      call chmalloc('sccdftbini.src','SCCTBINI','cgtmp',natom,crl=cgtmp)
      cgtmp(1:natom) = CG(1:natom)
  endif
  call SCCTBSEL(ISLCT,QQINP)
  if(qsccpert) then 
      call chmalloc('sccdftbini.src','SCCTBINI','charge1',natom,crl=charge1)
      ! charge1 is the charge array for lambda = 0
      charge1(1:natom) = CG(1:natom)
  endif

!  QC: 11/17 Done

  !     This is maybe not optimal???? I think it is!!!
#if KEY_REPLICA==1
#if KEY_RPATH==1
  IF(QREP) THEN
     CALL RECOPSEL(IREPQM)
     !C         CALL ENVIGRP(IREPQM)   
  ENDIF
#endif 
#endif 

  CALL MKMMLST

  CALL CHMSCCTB
  !
  !     ======================= Initilization step with DYLCAO ===========
  ! QC: 11/17 Add PERT based on Xiya
  ! if(qlamda.eqv..false.) then
  if((qlamda.eqv..false.) .and. (qsccpert .eqv. .false.)) then 
     !     ----- conventional single structure run

     CALL CHQSCCTB
     !
     !     This initializes sccdftb data, no energy calculations. 

#if KEY_PARALLEL==1
     MYSCCID=MYNOD
#else /* KEY_PARALLEL */
     MYSCCID=0
#endif 
     EXTFLAG='CH'
     qaflag=.false.
     qbflag=.false.
     CALL dylcao
     !
     !     QC: change 12/09
     call chmdealloc('sccdftbini.src','SCCTBINI','ISLCT',NATOM,intg=ISLCT)
     IF (QLINK) call chmdealloc('sccdftbini.src','SCCTBINI','LSLCT',NATOM,intg=LSLCT)

  ! QC: 11/17 Add PERT based on Xiya
  !ELSE 
   ELSE IF (qlamda) THEN 
     !
     ! =============== for lamda from guohui ====================

     nptc1=nptc

     ! from above we just get immlst and igmsel lists which are useful!!
     ! so we must get new parameters for nscctc for scctbene for A and B.
     ! so save immlst and igmsel to other temporary arrays.
     ! these two arrays just used for mm part calculations.
     ! but qm part must be seperated by iqmlst1 and iqmlst2,
     ! then copy one of them to be iqmlst to be used in scctbene.

     call chmalloc('sccdftbini.src','SCCTBINI','immlst1',maxa,intg=immlst1)
     call chmalloc('sccdftbini.src','SCCTBINI','igmsel1',maxa,intg=igmsel1)
     call chmalloc('sccdftbini.src','SCCTBINI','cptc1',3,maxptc,crl=cptc1)
     call chmalloc('sccdftbini.src','SCCTBINI','zptc1',maxptc,crl=zptc1)
     immlst1(1:maxa) = immlst(1:maxa)
     igmsel1(1:maxa) = igmsel(1:maxa)
     cptc1(1:3,1:maxptc) = cptc(1:3,1:maxptc)
     zptc1(1:maxptc) = zptc(1:maxptc)
     ! just for QM part
     do ii=1,maxa
        igmsel(ii)=0
     enddo

     call chmalloc('sccdftbini.src','SCCTBINI','ISLCT1',NATOM,intg=ISLCT1)
     call SELCTA(COMLYN,COMLEN,ISLCT1,X,Y,Z,WMAIN,.TRUE.)
     call SCCTBSEL(ISLCT1,QQINP)
     CALL CHQSCCTB
     nscctc1=nscctc
     do ii=1,nscctc1
        scctyp1(ii)=scctyp(ii)
        scczin1(ii)=scczin(ii)
        sccatm1(ii)=sccatm(ii)
     enddo
     do ii=1,nscctc1
        iqmlst1(ii,1)=iqmlst(ii,1)
        !           write(outu,*) 'nscctc1=',nscctc1,iqmlst1(ii,1)
     enddo

     !         temporary restore data of MM part for Dylcao.
     immlst(1:maxa) = immlst1(1:maxa)
     igmsel(1:maxa) = igmsel1(1:maxa)
     cptc(1:3,1:maxptc) = cptc1(1:3,1:maxptc)
     zptc(1:maxptc) = zptc1(1:maxptc)
     nptc=nptc1
     ichscc=ichscc1

#if KEY_PARALLEL==1
     MYSCCID=MYNOD
#else /* KEY_PARALLEL */
     MYSCCID=0
#endif 
     EXTFLAG='CH'
     qaflag=.true.
     qbflag=.false.
     lcolspin=lcolspin1 !MG_121101
     CALL dylcao
     !     ---------------------------------------------------------------
     do ii=1,maxa
        igmsel(ii)=0
     enddo

     call chmalloc('sccdftbini.src','SCCTBINI','ISLCT2',NATOM,intg=ISLCT2)
     call SELCTA(COMLYN,COMLEN,ISLCT2,X,Y,Z,WMAIN,.TRUE.)
     call SCCTBSEL(ISLCT2,QQINP)
     ichscc=ichscc2 ! MG_UW1211: for a proper output... (see subroutine CHQSCCTB)
     CALL CHQSCCTB
     nscctc2=nscctc
     do ii=1,nscctc2
        scctyp2(ii)=scctyp(ii)
        scczin2(ii)=scczin(ii)
        sccatm2(ii)=sccatm(ii)
     enddo
     do ii=1,nscctc2
        iqmlst2(ii,1)=iqmlst(ii,1)
     enddo
     !         restore immlst and igmsel arrays, and nptc, nprimm
     immlst(1:maxa) = immlst1(1:maxa)
     igmsel(1:maxa) = igmsel1(1:maxa)
     cptc(1:3,1:maxptc) = cptc1(1:3,1:maxptc)
     zptc(1:maxptc) = zptc1(1:maxptc)

     nptc=nptc1
     ichscc=ichscc2
     !
     !     This initialize scctb data, no energy calculations.
#if KEY_PARALLEL==1
     MYSCCID=MYNOD
#else /* KEY_PARALLEL */
     MYSCCID=0
#endif 
     EXTFLAG='CH'
     qbflag=.true.
     qaflag=.false.
     lcolspin=lcolspin2 !MG_121101
     CALL dylcao
     ! for getting other important information, you must read sub. dylcao!
     !
     !     QC: 12/09
     call chmdealloc('sccdftbini.src','SCCTBINI','ISLCT1',NATOM,intg=ISLCT1)
     call chmdealloc('sccdftbini.src','SCCTBINI','ISLCT2',NATOM,intg=ISLCT2)

     call chmdealloc('sccdftbini.src','SCCTBINI','immlst1',maxa,intg=immlst1)
     call chmdealloc('sccdftbini.src','SCCTBINI','igmsel1',maxa,intg=igmsel1)
     call chmdealloc('sccdftbini.src','SCCTBINI','cptc1',3,maxptc,crl=cptc1)
     call chmdealloc('sccdftbini.src','SCCTBINI','zptc1',maxptc,crl=zptc1)
  ! QC: 11/17 Add PERT based on Xiya   
  else if (qsccpert) then       !Xiya

     CALL CHQSCCTB

     ! get new parameters for nscctc for scctbene for A and B.
     ! igmsel part must be seperated by igmsel1 and igmsel2
     ! qm part must be seperated by iqmlst1 and iqmlst2,
     ! then copy one of them to be iqmlst to be used in scctbene.

     call chmalloc('sccdftbini.src','SCCTBINI','igmsel1',maxa,intg=igmsel1)
     call chmalloc('sccdftbini.src','SCCTBINI','charge2',natom,crl=charge2)
     igmsel1(1:maxa) = igmsel(1:maxa)

     nscctc1=nscctc
     do ii=1,nscctc1
        scctyp1(ii)=scctyp(ii)
        scczin1(ii)=scczin(ii)
        sccatm1(ii)=sccatm(ii)
     enddo
     do ii=1,nscctc1
        iqmlst1(ii,1)=iqmlst(ii,1)
     enddo

#if KEY_PARALLEL == 1
     MYSCCID=MYNOD
#else
     MYSCCID=0
#endif
     EXTFLAG='CH'
     qaflag=.true.
     qbflag=.false.
     lcolspin=lcolspin1 !MG_121101
     CALL dylcao
     !     ---------------------------------------------------------------
     do ii=1,maxa
        igmsel(ii)=0
     enddo

     call chmalloc('sccdftbini.src','SCCTBINI','ISLCT2',NATOM,intg=ISLCT2)
     call chmalloc('sccdftbini.src','SCCTBINI','igmsel1',maxa,intg=igmsel2)

     call SELCTA(COMLYN,COMLEN,ISLCT2,X,Y,Z,WMAIN,.TRUE.)

     ! copy original cg to CG array
     CG(1:natom)=cgtmp(1:natom)
     call SCCTBSEL(ISLCT2,QQINP)
     ! charge2 is the charge array for lambda=1
     charge2(1:natom)=CG(1:natom)

     CALL MKMMLST
     CALL CHMSCCTB

     igmsel2(1:maxa) = igmsel(1:maxa)
     ichscc=ichscc2 ! MG_UW1211: for a proper output... (see subroutine CHQSCCTB)

     CALL CHQSCCTB

     nscctc2=nscctc
     do ii=1,nscctc2
        scctyp2(ii)=scctyp(ii)
        scczin2(ii)=scczin(ii)
        sccatm2(ii)=sccatm(ii)
     enddo
     do ii=1,nscctc2
        iqmlst2(ii,1)=iqmlst(ii,1)
     enddo

     !
     !     This initialize scctb data, no energy calculations.
#if KEY_PARALLEL == 1
     MYSCCID=MYNOD
#else
     MYSCCID=0
#endif
     EXTFLAG='CH'
     qbflag=.true.
     qaflag=.false.
     lcolspin=lcolspin2 !MG_121101
     CALL dylcao

     call chmdealloc('sccdftbini.src','SCCTBINI','ISLCT2',NATOM,intg=ISLCT2)
     call chmdealloc('sccdftbini.src','SCCTBINI',"cgtmp",natom,crl=cgtmp)

  ! QC: 11/17 Done
  endif

  !     QC:UW_031205 make sure that QM group flag thas NOT been done
  !     used in nbondg.src
  QQMGRP=.false.
  !
#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1
#if KEY_PARALLEL==1 /*parallel*/
  !     Restore the parallel info
  !     
  IF(npgrp.ne.1)THEN
     MYNOD=MYNODG
     MYNODP=MYNOD+1
     NUMNOD=NUMNODG
     NODDIM=NPTWO()
     CALL CUBE(MYNOD,NUMNOD,IPPMAP)
  ENDIF
  !     
#endif /* (parallel)*/
#endif 
#endif /* (replica)*/
  !
  RETURN
END SUBROUTINE SCCTBINI
!
SUBROUTINE CHMSCCTB
  !-----------------------------------------------------------------------
  !     Define CHARMM atoms as point charges and COPY TO COMMON BLOCK
  !     to pass over to SCCDFTB. CALLED BY SCCTBENE.
  !
  !     Q. Cui, Xmas, 1998
  !
  use dimens_fcm
  use exfunc
  !
  use coord
  use psf
  !  use stackm (QC: corrected 12/09)
  use stream
  use sccdftb

  use sccdftbsrc, only: qsccnb,sccfnb,qsccs,qsccsh
  use sccdftb, only: igeometry

  use consta
  use gamess_fcm
  !
  implicit none

  INTEGER I

  !     NPTC = NTOTMM
  DO I = 1, NPTC
     CPTC(1,I) = X (IMMLST(I))/BOHRR
     CPTC(2,I) = Y (IMMLST(I))/BOHRR
     CPTC(3,I) = Z (IMMLST(I))/BOHRR
     !          MOVE THE CHARGE STUFF OVER TO MKMMLST
     !          ZPTC(I)   = CG(IMMLST(I))
  ENDDO

  !     Require transferring the data to the 2nd LST as well for 
  !     eWald/cut
  if (period.and.qsccnb) then
     DO I = 1, NPTR
        CPTR(1,I) = X (IMM2LST(I))/BOHRR
        CPTR(2,I) = Y (IMM2LST(I))/BOHRR
        CPTR(3,I) = Z (IMM2LST(I))/BOHRR
     ENDDO
  endif
  RETURN
END SUBROUTINE CHMSCCTB

!     XIAO_QC_UW0609: for SCC/eWald
SUBROUTINE getatomj(kmax,QSCCEWC,QSCCEWS)
  use chm_kinds
  use dimens_fcm
  !  use heapm QC: 12/09
  use sccdftb
  use consta
  use number
#if KEY_PARALLEL==1
  use parallel    
#endif
  use memory ! MG_QC_UW1206

  implicit none
  real(chm_real) QSCCEWC(*),QSCCEWS(*)
  real(chm_real) recbasis(3,3)
  real(chm_real) vol
  integer kmax !,kmaxalloc
  !parameter(kmaxalloc=20)

  ! MG_QC_UW1206 
  real(chm_real),allocatable,dimension(:,:) :: ktabxc,ktabxs,ktabyc,ktabys,ktabzc,ktabzs

  ! MG_QC_UW1205: bugfix
  ! REAL(chm_real) KTABXC(MAXPTC,0:KMAXALLOC),KTABXS(MAXPTC,0:KMAXALLOC)
  ! REAL(chm_real) KTABYC(MAXPTC,-KMAXALLOC:KMAXALLOC),KTABYS(MAXPTC,-KMAXALLOC:KMAXALLOC)
  ! REAL(chm_real) KTABZC(MAXPTC,-KMAXALLOC:KMAXALLOC),KTABZS(MAXPTC,-KMAXALLOC:KMAXALLOC)
  REAL(chm_real) KR1,KR2,KR3

  INTEGER I,J
  integer kx,ky,kz
  real(chm_real) ktg1,ktg2
  real(chm_real) ct,st

  INTEGER ISTART,IFINISH,NAT

  ! MG_QC_UW1206 
  call chmalloc('sccdftbini.src','getatomj','ktabxc',nptc,kmax+1,crl=ktabxc,lbou2=0)
  call chmalloc('sccdftbini.src','getatomj','ktabxs',nptc,kmax+1,crl=ktabxs,lbou2=0)
  call chmalloc('sccdftbini.src','getatomj','ktabyc',nptc,2*kmax+1,crl=ktabyc,lbou2=-kmax)
  call chmalloc('sccdftbini.src','getatomj','ktabys',nptc,2*kmax+1,crl=ktabys,lbou2=-kmax)
  call chmalloc('sccdftbini.src','getatomj','ktabzc',nptc,2*kmax+1,crl=ktabzc,lbou2=-kmax)
  call chmalloc('sccdftbini.src','getatomj','ktabzs',nptc,2*kmax+1,crl=ktabzs,lbou2=-kmax)

  ! MG_QC_UW1205: extra check  
  ! if (kmax>kmaxalloc) then
  !  write(*,*) 
  !  write(*,*) 'ERROR: parameter --kmaxalloc-- too small! '
  !  write(*,*) 
  !  write(*,*) 'Possible Solutions: '
  !  write(*,*) '1. lower kmax in input or'
  !  write(*,*) '2. change --kmaxalloc-- to a larger number within' 
  !  write(*,*) '   CHARMMCODE/source/sccdftbint/sccdftbini.src and recompile'
  !  write(*,*) 
  !  stop
  ! endif

#if KEY_PARALLEL==1
  ISTART=1+IKVCSC(MYNOD)
  IFINISH= IKVCSC(MYNODP)
#else /* KEY_PARALLEL */
  ISTART=1
  IFINISH=NKVEC
#endif 

  CALL REZVOL(boxsiz,recbasis,vol)
  !     write(*,*) "getatomj>",nkvec,kmax,boxsiz(1,1),boxsiz(2,2),boxsiz(3,3),nptc

  DO I=1,NPTC
     !     construct cos and sin table for each atom(MM)
     KTABXC(I,0) = ONE
     KTABXS(I,0) = ZERO
     !
     KTABYC(I,0) = ONE
     KTABYS(I,0) = ZERO
     !
     KTABZC(I,0) = ONE
     KTABZS(I,0) = ZERO
     KR1 = recbasis(1,1)*cptc(1,I) + & 
          recbasis(2,1)*cptc(2,I) + & 
          recbasis(3,1)*cptc(3,I)

     KR2 = recbasis(1,2)*cptc(1,I) + & 
          recbasis(2,2)*cptc(2,I) + & 
          recbasis(3,2)*cptc(3,I)

     KR3 = recbasis(1,3)*cptc(1,I) + & 
          recbasis(2,3)*cptc(2,I) + & 
          recbasis(3,3)*cptc(3,I)

     KTABXC(I,1) = COS(KR1)
     KTABXS(I,1) = SIN(KR1)
     KTABYC(I,1) = COS(KR2)
     KTABYS(I,1) = SIN(KR2)
     KTABYC(I,-1) =  KTABYC(I,1)
     KTABYS(I,-1) = -KTABYS(I,1)
     KTABZC(I,1) = COS(KR3)
     KTABZS(I,1) = SIN(KR3)
     KTABZC(I,-1) =  KTABZC(I,1)
     KTABZS(I,-1) = -KTABZS(I,1)


     DO J = 2, KMAX
        KTABXC(I,J)=KTABXC(I,1)*KTABXC(I,J-1) &
             -KTABXS(I,1)*KTABXS(I,J-1)
        KTABXS(I,J)=KTABXS(I,1)*KTABXC(I,J-1) &
             +KTABXC(I,1)*KTABXS(I,J-1)
     ENDDO
     DO J = 2, KMAX
        KTABYC(I,J)=KTABYC(I,1)*KTABYC(I,J-1) &
             -KTABYS(I,1)*KTABYS(I,J-1)
        KTABYS(I,J)=KTABYS(I,1)*KTABYC(I,J-1) &
             +KTABYC(I,1)*KTABYS(I,J-1)
        KTABYC(I,-J)= KTABYC(I,J)
        KTABYS(I,-J)=-KTABYS(I,J)
     ENDDO
     DO J = 2, KMAX
        KTABZC(I,J)=KTABZC(I,1)*KTABZC(I,J-1) &
             -KTABZS(I,1)*KTABZS(I,J-1)
        KTABZS(I,J)=KTABZS(I,1)*KTABZC(I,J-1) &
             +KTABZC(I,1)*KTABZS(I,J-1)
        KTABZC(I,-J)= KTABZC(I,J)
        KTABZS(I,-J)=-KTABZS(I,J)
     ENDDO
  ENDDO

  DO J=ISTART,IFINISH
     KX = KXV(J)
     KY = KYV(J)
     KZ = KZV(J)
     QSCCEWC(J)=ZERO
     QSCCEWS(J)=ZERO
     !      write(*,*) "getatomj> kx ",kx,ky,kz 

     DO I=1,NPTC
        KTG1 = KTABZC(I,KZ)*KTABYC(I,KY)-KTABYS(I,KY)*KTABZS(I,KZ)
        KTG2 = KTABZC(I,KZ)*KTABYS(I,KY)+KTABYC(I,KY)*KTABZS(I,KZ)
        CT = zptc(i)*(KTABXC(I,KX)*KTG1 - KTABXS(I,KX)*KTG2)
        ST = zptc(i)*(KTABXS(I,KX)*KTG1 + KTABXC(I,KX)*KTG2)
        QSCCEWC(J)=QSCCEWC(J) + ct
        QSCCEWS(J)=QSCCEWS(J) + st
     ENDDO
  ENDDO

  !     write(*,*) "getatomj> qsccewc",qsccewc(1),qsccews(2),zptc(1),zptc(2),zptc(3)

  ! MG_QC_UW1206 
  call chmdealloc('sccdftbini.src','getatomj','ktabxc',nptc,kmax+1,crl=ktabxc)
  call chmdealloc('sccdftbini.src','getatomj','ktabxs',nptc,kmax+1,crl=ktabxs)
  call chmdealloc('sccdftbini.src','getatomj','ktabyc',nptc,2*kmax+1,crl=ktabyc)
  call chmdealloc('sccdftbini.src','getatomj','ktabys',nptc,2*kmax+1,crl=ktabys)
  call chmdealloc('sccdftbini.src','getatomj','ktabzc',nptc,2*kmax+1,crl=ktabzc)
  call chmdealloc('sccdftbini.src','getatomj','ktabzs',nptc,2*kmax+1,crl=ktabzs)

  RETURN
END SUBROUTINE getatomj

!     Pre-compute eWald related terms 
!     QC_UW031205 (updated the next two subroutines for para)

SUBROUTINE SCCEWCS(QSCCEWC,QSCCEWS)
  use dimens_fcm
  use psf
  use sccdftb
  use gamess_fcm
  use consta
  use coord
  use number
  use parallel
  implicit none

  real(chm_real) QSCCEWC(*),QSCCEWS(*)

  INTEGER I,J,ITMP,ISTART,IFINISH
  real(chm_real)  TMP,TMPC,TMPS

#if KEY_PARALLEL==1
  ISTART=1+IKVCSC(MYNOD)
  IFINISH= IKVCSC(MYNODP)
#else /* KEY_PARALLEL */
  ISTART=1
  IFINISH=NKVEC
#endif 
  !     DO I=1,NKVEC
  DO I=ISTART,IFINISH

     QSCCEWC(I)=ZERO
     QSCCEWS(I)=ZERO
     DO J=1,NPTC
        itmp=IMMLST(J)
        tmp =KXVEC(I)*X(ITMP)+KYVEC(I)*Y(ITMP)+KZVEC(I)*Z(ITMP)
        tmpc=cos(tmp/BOHRR)
        tmps=sin(tmp/BOHRR)
        QSCCEWC(I)=QSCCEWC(I) + CG(ITMP)*tmpc
        QSCCEWS(I)=QSCCEWS(I) + CG(ITMP)*tmps
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE SCCEWCS

SUBROUTINE SCQEWCS(QSCQEWC,QSCQEWS,qzero,izp)
  use dimens_fcm
  use psf
  use sccdftb
  use gamess_fcm
  use consta
  use coord
  use number
  use parallel
  use sccdftbsrc, only: qmat,maxtyp
  implicit none

  real(chm_real) QSCQEWC(*),QSCQEWS(*)
  real(chm_real) qzero(maxtyp,4) !MG_UW1210 (ldep)
  INTEGER IZP(*)

  INTEGER I,J,ITMP,izpj,ISTART,IFINISH
  real(chm_real)  TMP,TMPC,TMPS

#if KEY_PARALLEL==1
  ISTART=1+IKVCSC(MYNOD)
  IFINISH= IKVCSC(MYNODP)
#else /* KEY_PARALLEL */
  ISTART=1
  IFINISH=NKVEC
#endif 
  DO I=1,NKVEC
     QSCQEWC(I)=ZERO
     QSCQEWS(I)=ZERO
  ENDDO

  DO I=ISTART,IFINISH
     !       QSCQEWC(I)=ZERO
     !       QSCQEWS(I)=ZERO
     DO J=1,NSCCTC
        itmp=IQMLST(J,1)
        izpj = izp(j)
        tmp =KXVEC(I)*X(ITMP)+KYVEC(I)*Y(ITMP)+KZVEC(I)*Z(ITMP)
        tmpc=cos(tmp/BOHRR)
        tmps=sin(tmp/BOHRR)
        !         ...... Note the sign here ...... 
        QSCQEWC(I)=QSCQEWC(I) - (qmat(j)-qzero(izpj,4))*tmpc
        QSCQEWS(I)=QSCQEWS(I) - (qmat(j)-qzero(izpj,4))*tmps
        !         if (I.eq.1) then
        !           write(*,*) "qmat> scqewcs: ",j,qmat(j),qzero(izpj,4)
        !         endif
     ENDDO
  ENDDO

  !     We need to make sure that all nodes have the same value because
  !     ALL values are used in computing the forces (we distribute things
  !     differently)
#if KEY_PARALLEL==1
  CALL GCOMB(QSCQEWC,NKVEC)
  CALL GCOMB(QSCQEWS,NKVEC)
#endif 
  !     Should not have to do anything else 'cause we have the same node
  !     keeping the same set of data
  RETURN  
END SUBROUTINE SCQEWCS


SUBROUTINE CHQSCCTB
  !----------------------------------------------------------------------
  !     Find the atoms defined as QM atoms and get them ready
  !     for SCCDFTB. 
  !
  !     Q. Cui, Xmas, 1998
  !
  use dimens_fcm
  use number
  use exfunc
  !
  use coord
  use param
  use psf
  use rtf,only:atct
  use stream
  use sccdftb
  use gamess_fcm
  use linkatom, only: findel
  ! added for regroup GHO at end of QM, 0601PJ07
  use stbgho
  !
  implicit none

  !     QC: May, 2001: Increase AATOM,AZNUC for replica

  CHARACTER(len=10) AATOM(MAXCEN*3)
  real(chm_real) AZNUC(MAXCEN*3)
  INTEGER NAT
  !
  ! remove NATQM for a moment to avoid conflict with stbgho.f90, 0601PJ07
  !     INTEGER I,N,NSLCT,NATMM,NATQM,NATLNK
  INTEGER I,N,NSLCT,NATMM,NATLNK
  CHARACTER(len=6) ELE
  LOGICAL QPRT
  INTEGER KSCCRP,MSCC

  ! local variables added, 0601PJ07
  INTEGER J,K
  LOGICAL QGHO
  !
  !     QC: no change for image.

  QPRT=.TRUE.
  N=0
  DO I = 1,NATOM
     ! skip GHO atoms for a moment, 0601PJ07
     QGHO = .FALSE.
     IF (QLINK .AND. NQMLNK .GE. 1) THEN
        DO J = 1, NQMLNK
           K = IQLINK(J)
           IF (K .EQ. I) QGHO = .TRUE.
        ENDDO
     ENDIF
     IF (((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)).AND..NOT.QGHO) THEN
        !        IF ((IGMSEL(I) .EQ. 1).OR.(IGMSEL(I).EQ.2)) THEN
        N = N + 1
        AATOM(N)=ATYPE(I)
        IF (ATYPE(I)(1:3) == 'QQH') AATOM(N)=' H'
        !
        CALL FINDEL(ATCT(IAC(I)),AMASS(I),I,ELE,AZNUC(N),QPRT)
        !
        SCCATM(N)=AATOM(N)
        SCCZIN(N)=AZNUC(N)
        !           QC: WE TAKE THE ATOM TYPE FROM THE WMAIN
        SCCTYP(N)=INT(WMAIN(I))
     ENDIF
  ENDDO

  ! pack GHO atoms at the end of the QM fragment, 0601PJ07
  IF (QLINK .AND. NQMLNK .GE. 1) THEN
     DO J = 1, NQMLNK
        I = IQLINK(J)
        N = N + 1
        AATOM(N)=ATYPE(I)
        CALL FINDEL(ATCT(IAC(I)),AMASS(I),I,ELE,AZNUC(N),QPRT)
        SCCATM(N)=AATOM(N)
        SCCZIN(N)=AZNUC(N)
        SCCTYP(N)=INT(WMAIN(I))
        !SCCMASS(N)=AMASS(I)
     ENDDO
  ENDIF
  !
  NAT=N
  !
  IF (NAT .LE. 0) CALL WRNDIE(0,'<CHQSCC>', &
       'No quantum mechanical atoms selected.')
  NATMM = NATOM -N
  NATQM = N
  NSCCTC = N
  !
  NSLCT = 0
  DO I = 1,NATOM
     IF (IGMSEL(I).EQ.2) NSLCT = NSLCT + 1
  ENDDO
  NATLNK = NSLCT
  !
  !     Write out atomic information
  !
#if KEY_PARALLEL==1
  IF (MYSCCID.EQ.0) THEN 
#endif 
     IF (PRNLEV.GT.2) WRITE (OUTU,'(/,1x,A,/)') &
          ' SCCDFN> Some atoms will be treated quantum mechanically.'
     IF (PRNLEV.GT.2) WRITE (OUTU,'(4(8X,A,I5,/),/)') &
          ' The number of SCCDFTB        QM  atoms   = ',NATQM, &
          ' The number of molecular mechanical atoms = ',NATMM, &
          ' The number of MM atoms excluded from QM  = ',NATMM-NPTC, &
          ' Of which the number of QM/MM link atoms  = ',NATLNK 

     WRITE (OUTU,*) "CHARGE OF SCCDFTB ATOMS: ",ICHSCC
#if KEY_PARALLEL==1
  ENDIF
#endif 
  !     Deal with replication 
  !     QC: UW0110: the following SHOULD BE quoted out -->  
  !     although one has to be careful about compatability
  !     with RPATH later @@@ Confirm with LEE Woodcock
  IF (LSCCRP) THEN
     NSCCTC=NSCCTC/NSCCRP

#if KEY_PARALLEL==1
     IF (MYSCCID.EQ.0) THEN 
#endif 
        WRITE(OUTU,'(1x,"SCCTB Replicated for ",I5," times.")') NSCCRP
        WRITE(OUTU,'(1x,"SCCTB Atoms in each replica: ",I5)') NSCCTC
#if KEY_PARALLEL==1
     ENDIF
#endif 
  ENDIF

  !     QC: Make up a QM list 
  N=0
  DO I = 1,NATOM
     ! skip GHO atoms for a moment, 0601PJ07
     QGHO = .FALSE.
     IF (QLINK .AND. NQMLNK .GE. 1) THEN
        DO J = 1, NQMLNK
           K = IQLINK(J)
           IF (K .EQ. I) QGHO = .TRUE.
        ENDDO
     ENDIF
     IF (((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)).AND..NOT.QGHO) THEN
        !        IF ((IGMSEL(I) .EQ. 1).OR.(IGMSEL(I).EQ.2)) THEN
        N = N + 1
        KSCCRP=INT((N-1)/NSCCTC) + 1
        MSCC  =N - (KSCCRP-1)*NSCCTC 
        IQMLST(MSCC,KSCCRP) = I
     ENDIF
  ENDDO

  ! pack GHO atoms at the end of the QM fragment, 0601PJ07
  IF (QLINK .AND. NQMLNK .GE. 1) THEN
     DO J = 1, NQMLNK
        I = IQLINK(J)
        N = N + 1
        KSCCRP=INT((N-1)/NSCCTC) + 1
        MSCC  =N - (KSCCRP-1)*NSCCTC
        IQMLST(MSCC,KSCCRP) = I
     ENDDO
  ENDIF

  !     Update CMD XYZ 

#if KEY_PARALLEL==1
  IF (MYSCCID.EQ.0) THEN 
#endif 
     IF (LSCCRP.AND.LCNSCD) CALL UPCMDXYZ(X,Y,Z,PRNLEV,OUTU)
#if KEY_PARALLEL==1
  ENDIF
#endif 

  RETURN
END SUBROUTINE CHQSCCTB
!
SUBROUTINE SCCTBENE(CTOT,X,Y,Z,DX,DY,DZ,NATOM,lgrad)
  !-----------------------------------------------------------------------
  !
  !     Get energy and forces from SCCDFTB
  !
  !     Q. Cui, Xmas, 1998.
  !
  use dimens_fcm
  use exfunc
  use blockscc_fcm
  !
  ! added for GHO-SCC-DFTB ... PJ 7/2004
  use stbgho
  use block_fcm        
  use consta
#if KEY_PARALLEL==1
  use econtmod 
#endif
  use gamess_fcm
  !  use heapm QC: 12/09
  use memory !QC: 12/09
  use number
  use parallel     
  use sccdftb
  use sccdftbsrc, only: LSCCDIP, UDIP, qsccnb,sccfnb,qsccs,qsccsh,qsccpme, &
#if KEY_DFTBPLUS==1
      ldftbplus, dftb_external_charges, dftb_klopman_ohno, dftb_gsbp_pot, &
      dftb_plus, &
#endif
  lsccdftd2,lsccdftd3,lcpe,lcdko


  ! PZ MG_QC_UW1206
  use stream
  !     QC_UW04
  use image        
  !     XIAO_QC_UW0609: pass ewald info
  use erfcd_mod, only: EWNPTS,EWLDT,EWRDEL 
  ! use dftd3, only: calc_e_dftd3, calc_g_dftd3
  !  use ewald QC: need to remove 12/09; instead copy from erfcd.src
  !   EWNPTS       - number of points on the erfc lookup table
  !   EWLDT        - heap pointer for the erfc lookup table
  !   EWRDEL       - reciprocal of spacing on the erfc lookup table
  ! real(chm_real),allocatable,dimension(:),save :: EWLDT
  ! real(chm_real),save :: ewrdel
  ! integer,save :: ERFMOD,EWNPTS
  ! ----------------------------------

#if KEY_DFTBPLUS==1
  use external_charges, only: ExternalCharges, updateExtChargePotCoords=>updateExtPotCoords
  use dftbplus_interface, only: calculate_dftb_plus, setup_dftb_keywords
  use klopman_ohno, only: updateKlopmanOhnoPotCoords=>updateExtPotCoords
  use charmm_dftb, only: CharmmDftbInput
#endif

  implicit none

  real(chm_real),allocatable,dimension(:) :: IPHO
  real(chm_real),allocatable,dimension(:) :: IFAO
  real(chm_real),allocatable,dimension(:) :: ISAO
  real(chm_real),allocatable,dimension(:) :: IWHO
  real(chm_real),allocatable,dimension(:) :: IFHB
  real(chm_real),allocatable,dimension(:) :: ISHB
  real(chm_real),allocatable,dimension(:) :: ICHB
  real(chm_real),allocatable,dimension(:) :: ICAO


  real(chm_real) ECLASS
  !
  real(chm_real) CTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  real(chm_real) GTOT,GTOTOLD,ECURRENT
  INTEGER NATOM,N1
  !
  !     QC_UW04
  !
  !     QC: We make a pseudo-parallel SCC, i.e., only the main
  !     node calls the SCCTBENE - this is useful because when the 
  !     number of MM atoms becomes large, the cost could overshadow
  !     SCC calculations (with PARALLEL).
  !     The PARASCC option gives the parallel version for path-integral
  !     or replica path

#if KEY_PARASCC==1
  INTEGER NTRY,ISTART,IFINISH
  INTEGER IRPLSCC(256)  
#endif 

  INTEGER ICHARM
  real(chm_real) C(3,MAXCEN),E,EEL,EG(3,MAXCEN)
  !
  !     QC: Add dispersion here (moved to eglcao). 

  !c     INTEGER MAXTYP
  !c     parameter( MAXTYP= 6) 
  !c     real(chm_real) Edis
  !c     LOGICAL dispers
  !c     common /disper/ Edis,dispers
  !
  INTEGER I,N,NTOT,KK,IBL,IBLQM
  INTEGER ITER
  real(chm_real)  KSCPINT
  !     QC_UW0406: add eWald related
  INTEGER NKTMP
  !     QC: 12/09
  real(chm_real),allocatable,dimension(:) :: ISCCEWC
  real(chm_real),allocatable,dimension(:) :: ISCCEWS
  real(chm_real),allocatable,dimension(:) :: ISCQEWC
  real(chm_real),allocatable,dimension(:) :: ISCQEWS
  !
  !  added for GHO ... PJ 7/2004 
  !     INTEGER IPHO,IFAO,ISAO,IWHO,IFHB,ISHB,ICHB,ICAO,
  INTEGER NGLT,NGSQ   
  !
  !  XIAO_QC_UW0609 force flag
  LOGICAL lgrad

  ! ASC: Dispersion energy contribution
  real(chm_real) e_dftd3_e
  real*8 e_dftd3_inout

  ! ASC: Dispersion gradient contribution. Defined this way to 
  ! fit into the wrapper for Grimmes code.
  real*8, dimension(3,20000) :: g_dftd3
  real*8 :: g_dftd3_norm

#if KEY_DFTBPLUS==1
  type(CharmmDftbInput) :: setup
  real(chm_real) :: eplus
  real(chm_real), allocatable :: gplus(:,:)
  real(chm_real), allocatable :: chargesplus(:)
#endif

#if KEY_DFTBPLUS==1
    ! ASC: DFTB+ initialization
    if ((ldftbplus).and.(dftb_plus%initialized.eqv..false.)) then 
        call setup_dftb_keywords(setup)
    endif
#endif

  if(qlamda.eqv..false.) scal=1.0d0

  !     write(OUTU,*) "Getting  here???",NSCCTC
  IF(NSCCTC.EQ.0) RETURN
  KSCPINT=ONE /DBLE(NSCCRP)

  !     QC:UW_041705 DEBUG
  if (prnlev.gt.6)      &
       write(OUTU,'(1x, "Scaling factor for QM/MM: ",F10.5)') KSCPINT
  CTOT=ZERO
  NTOT=NPTC
  GTOT=ZERO
  GTOTOLD=ZERO
  !     IF ((.NOT.LQMPB).and.(qsccnb.eqv..false.)) NTOT=NPRIMM
  !     IF (qsccnb.eqv..false.) NTOT=NPRIMM

  !     ---------------------------------------------------------------
  !     QC_UW04
  !     in case of periodic boundary condition, we see if necessary to
  !     update the lattice the QM atoms see - for eWald summation
  !     we chose to update nlat etc at the same time - perhaps not necc
  if (period.and.iuptbox.ne.0)  &
       call updbox(boxsiz,XTLABC,1,NLAT)
  !     QC_UW04
  !     ---------------------------------------------------------------

  !     ====================== Iteration here for replica ===============

#if KEY_PARASCC==0
  !     &&&&&&&&&&&&&&&&&&&& First a non-parallel version &&&&&&&&&&&&&&&&
  !

  BLFACTOR=ONE
#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1
#if KEY_PARALLEL==1 /*parallel*/

  !     determine block factor (for every QM package)

#if KEY_BLOCK==1 /*block*/
  IF((.not. qlamda).and.(.not. qpkac))then !Puja
  !        WRITE(*,*)'QBLOCK= ',QBLOCK
  IF(QBLOCK)THEN
     IBLQM=0
     DO I=1,NATOM
        !           WRITE(*,*)'BLOCKFACTOR= ',BLFACTOR
        IBL=IBLCKP(I)
        IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
           IF(IBLQM.EQ.0)IBLQM=IBL
           IF(IBLQM.NE.IBL)THEN
              CALL WRNDIE(-5,'<SCCTBENE>', &
                   'QM region must be within the same block.')
           ELSE
              KK=IBL*(IBL+1)/2
              BLFACTOR=BLCOEB(KK)
           ENDIF
        ENDIF
     ENDDO
  ENDIF
  ENDIF !Puja
#endif /*  (block)*/

  IF((.not. qlamda).and.(.not. qpkac))then !Puja
  IF(NPGRP .NE. 1)then

     MYNOD=mynod_save
     MYNODP=MYNOD+1
     NUMNOD=numnod_save
     NODDIM=NPTWO()

     DO I=0,NUMNOD
        IPPMAP(I)=I+KREPQM*(IREPQM-1)
     ENDDO
     !         
     IF(PRNLEV.GT.2) THEN
        IF(OPTCNT.EQ.0) THEN
           WRITE(OUTU,*)'SCCTBENE> USE ',npgrp, &
                ' GROUPS OF PARALLEL PROCESSES'
           WRITE(OUTU,*)
        ENDIF
        OPTCNT=OPTCNT+1
     ENDIF
  ENDIF
  ENDIF !Puja
#endif /* (parallel)*/
#endif 
#endif /* (replica)*/

  DO 777 ITER =1,NSCCRP

     ! Update coordinates for continuous calculation for SCCDFTB

     DO I = 1,NSCCTC
        C(1,I)=X(IQMLST(I,ITER))/BOHRR
        C(2,I)=Y(IQMLST(I,ITER))/BOHRR
        C(3,I)=Z(IQMLST(I,ITER))/BOHRR
     ENDDO

     ! Update the MM part as well.

     CALL CHMSCCTB

     IF(NPTC.NE.0) THEN
        DO ICHARM=1,NPTC
           DXTBMM(ICHARM)=ZERO
           DYTBMM(ICHARM)=ZERO
           DZTBMM(ICHARM)=ZERO
        ENDDO
     ENDIF
     !    QC_UW031205
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif 
        if (period.and.qsccnb) then
           DO ICHARM=1,NPTR
              DXTBM2(ICHARM)=ZERO
              DYTBM2(ICHARM)=ZERO
              DZTBM2(ICHARM)=ZERO
           ENDDO
        endif
        !    QC_UW031205
#if KEY_PARALLEL==1
     ENDIF
#endif 
     !     QC: UW04_06: in the case of ewald - as far as we do not use
     !     EOPT, then we pre-compute terms related to the MM charges which
     !     do NOT depend on the QM charges 
     NKTMP=1
     if (period.and..not.LSCOEWD) NKTMP=NKVEC

     call chmalloc('sccdftbini.src','SCCTBENE','ISCCEWC',NKTMP,crl=ISCCEWC)
     call chmalloc('sccdftbini.src','SCCTBENE','ISCCEWS',NKTMP,crl=ISCCEWS)
     call chmalloc('sccdftbini.src','SCCTBENE','ISCQEWC',NKTMP,crl=ISCQEWC)
     call chmalloc('sccdftbini.src','SCCTBENE','ISCQEWS',NKTMP,crl=ISCQEWS)

     !     QC: Allocate space for the similar quantities for QM atoms
     !     as well to facilitate derivative calculations involving ewald

     !    XIAO_QC_UW0609: ewald table
     !    PZ MG_QC_UW1206: PME
     if(.not.qsccpme) then
     if (period.and..not.LSCOEWD) &
          call getatomj(kmxscc,isccewc,isccews)
     endif ! qsccpme

     !
     ! allocate space for GHO large arrays ... PJ 7/2004
     !
     IF (QLINK) THEN
        CALL GTDIM(NSCCTC,NGLT,NGSQ)
     ELSE
        NGLT = 1
        NGSQ = 1
     ENDIF
     call chmalloc('sccdftbini.src','SCCTBENE','IPHO',NGLT,crl=IPHO)
     call chmalloc('sccdftbini.src','SCCTBENE','IFAO',NGLT,crl=IFAO)
     call chmalloc('sccdftbini.src','SCCTBENE','ISAO',NGLT,crl=ISAO)
     call chmalloc('sccdftbini.src','SCCTBENE','IWHO',NGLT,crl=IWHO)
     call chmalloc('sccdftbini.src','SCCTBENE','IFHB',NGLT,crl=IFHB)
     call chmalloc('sccdftbini.src','SCCTBENE','ISHB',NGLT,crl=ISHB)
     call chmalloc('sccdftbini.src','SCCTBENE','ICHB',NGSQ,crl=ICHB)
     call chmalloc('sccdftbini.src','SCCTBENE','ICAO',NGSQ,crl=ICAO)
     !
     ! update the hybridization matrix for GHO ... PJ 7/2004
     !
     IF(QLINK) CALL HBDEF(X,Y,Z,BT,BTM,DBTMMM,NQMLNK, &
          IQLINK,JQLINK,KQLINK)

#if KEY_PARALLEL==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
     GTOTOLD=GTOT 
#endif 
#endif 
#endif 

#if KEY_DFTBPLUS==1
    if (ldftbplus.eqv..false.) then
#endif
     ! Guishan_ZHENG 03/2013 (Initial guess option)
     CALL eglcao(nscctc*3,      c,       .false.,e,eel,eg, &
          lgrad, &
          isccewc,isccews,iscqewc,iscqews, &
          ewldt,ewrdel,ewnpts, &
          ! GHO ... PJ 7/2004
          ipho,ifao,isao,iwho, &
          ifhb,ishb,ichb,icao)  
#if KEY_DFTBPLUS==1
    else
      allocate(gplus(3,nndim))
      allocate(chargesplus(nndim))

      ! eg = 0.0d0
      ! eg = 0.0d0

      ! Update the QM-coordinates for the external potentials
      if ((extflag.eq.'CH').and.(.not.lcdko)) then
        call updateExtChargePotCoords(dftb_external_charges, c)

      else if ((extflag.eq.'CH').and.lcdko) then          
        call updateKlopmanOhnoPotCoords(dftb_klopman_ohno, c)

      endif

      call calculate_dftb_plus(nscctc, c, eplus, gplus, chargesplus)

      e = eplus

      qmulik(:nscctc) = - chargesplus(:nscctc)

      if (lgrad) then
          eg(:3,:nscctc) = - gplus(:3,:nscctc)
      else
          eg(:3,:nscctc) = 0.0d0
      endif

      deallocate(gplus)
      deallocate(chargesplus)

    endif

#endif

#if KEY_PARALLEL==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
     IF(MYNOD.EQ.0) THEN
        GTOT=E*TOKCAL*BLFACTOR
     ELSE
        GTOT=ZERO
     ENDIF
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
#endif /* */
#endif /* */
#endif /* */
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCCEWC',NKTMP,crl =ISCCEWC)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCCEWS',NKTMP,crl =ISCCEWS)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCQEWC',NKTMP,crl =ISCQEWC)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCQEWS',NKTMP,crl =ISCQEWS)

     !     -----------------------------------------------------------
     !     QC_UW04: Add muliken if asked, we transfer over to CG in
     !     energy.src after the nonbond terms are calculated.
     IF (LMULIK) THEN
        DO I=1,NSCCTC
           QMULI2(I,ITER)=QMULIK(I)
        ENDDO
     ENDIF
     !     Haibo_QC_UW0609: output the SCC-DFTB dipole moment
     IF (LSCCDIP) THEN
        call outputmullcoor(udip,nscctc,c,qmulik)
     ENDIF

     !     -----------------------------------------------------------
     !     QC_UW031205
#if KEY_PARALLEL==1
     !     THE FOLLOWING NEEDS TO BE DONE FOR ONLY THE HEADNODE
     IF (MYNOD.EQ.0) THEN
#endif 
        CTOT=CTOT + E*TOKCAL*KSCPINT*blfactor

        DO I=1,NSCCTC
           DX(IQMLST(I,ITER))=DX(IQMLST(I,ITER)) &
                +scal*EG(1,I)*TOKCAL/BOHRR*KSCPINT*blfactor
           DY(IQMLST(I,ITER))=DY(IQMLST(I,ITER)) &
                +scal*EG(2,I)*TOKCAL/BOHRR*KSCPINT*blfactor
           DZ(IQMLST(I,ITER))=DZ(IQMLST(I,ITER)) &
                +scal*EG(3,I)*TOKCAL/BOHRR*KSCPINT*blfactor

           IF(PRNLEV.GT.6) THEN
              WRITE(OUTU,'(I5,3F14.6)') I,EG(1,I)*TOKCAL/BOHRR     &
                   , EG(2,I)*TOKCAL/BOHRR &
                   , EG(3,I)*TOKCAL/BOHRR
           ENDIF
        ENDDO
        !
        !     MM atoms, without igmsel(i)=-1 !!
        !     QC: CHANGE TO ACCOUT FOR RCUT
        !     QC: YET WE SHOULD ZERO OUT THE QM/MM IMAGE FORCE ON THE IMAGE
        !     ATOM IF DOING INFINITELY DILUTE SYSTEM

        DO I=1,NTOT
           DX(IMMLST(I))=DX(IMMLST(I)) &
                +scal*DXTBMM(I)*TOKCAL/BOHRR*KSCPINT*blfactor
           DY(IMMLST(I))=DY(IMMLST(I)) &
                +scal*DYTBMM(I)*TOKCAL/BOHRR*KSCPINT*blfactor
           DZ(IMMLST(I))=DZ(IMMLST(I)) &
                +scal*DZTBMM(I)*TOKCAL/BOHRR*KSCPINT*blfactor
           ! Haibo Yu
           !      WRITE(*,*)ITER,I,DXTBMM(I)
        ENDDO
        !     QC: Need to add the real space part for eWald - MM
        if (period.and.qsccnb) then
           DO I=1,NPTR
              DX(IMM2LST(I))=DX(IMM2LST(I)) &
                   +scal*DXTBM2(I)*TOKCAL/BOHRR*KSCPINT
              DY(IMM2LST(I))=DY(IMM2LST(I)) &
                   +scal*DYTBM2(I)*TOKCAL/BOHRR*KSCPINT
              DZ(IMM2LST(I))=DZ(IMM2LST(I)) &
                   +scal*DZTBM2(I)*TOKCAL/BOHRR*KSCPINT
           ENDDO
        endif
        !     QC_UW031205
#if KEY_PARALLEL==1
     ELSE
        !     Other nodes do NOT know about the force or energy
        CTOT=ZERO
     ENDIF
#endif 

     !
     ! Determine the GHO Q-link atom derivatives ... PJ 7/2004
     !
     IF(QLINK) THEN
        CALL DQLINK(DX,DY,DZ,X,Y,Z,IPHO,IFAO)
        CALL DQLINK(DX,DY,DZ,X,Y,Z,IWHO,ISAO)
        ECLASS = 0.0D0
        CALL RPMMM(DX,DY,DZ,X,Y,Z,ECLASS)
        CTOT = CTOT + ECLASS
     ENDIF
     !
     ! free space required for GHO gradient calculations ... PJ 7/2004
     !
     call chmdealloc('sccdftbini.src','SCCTBENE','IPHO',NGLT,crl=IPHO)
     call chmdealloc('sccdftbini.src','SCCTBENE','IFAO',NGLT,crl=IFAO)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISAO',NGLT,crl=ISAO)
     call chmdealloc('sccdftbini.src','SCCTBENE','IWHO',NGLT,crl=IWHO)
     call chmdealloc('sccdftbini.src','SCCTBENE','IFHB',NGLT,crl=IFHB)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISHB',NGLT,crl=ISHB)
     call chmdealloc('sccdftbini.src','SCCTBENE','ICHB',NGSQ,crl=ICHB)
     call chmdealloc('sccdftbini.src','SCCTBENE','ICAO',NGSQ,crl=ICAO)

     !  QC: UW0110: Moved GHO related stuff inside the SCCRP Loop - CHECK PLEASE@@
     !     ^^^^^ Iteration end for replica
777 ENDDO
  !
#if KEY_PARALLEL==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
  !     
  !     Leave parallel/parallel mode
  !     
  IF(npgrp .ne. 1)THEN
     NUMNOD=NUMNODG
     MYNOD=MYNODG
     MYNODP=MYNOD+1
     NODDIM=NPTWO()
     CALL CUBE(MYNOD,NUMNOD,IPPMAP)
  ENDIF
#endif 
#endif 
#endif 

  !     ^^^^^ Iteration end for replica
#else /* ? */
  !     &&&&&&&&&&&&&&&&&& Now a parallel version &&&&&&&&&&&&&&&&&&&&&&&
  !     First make a list of nodes -- can move to other places
  IRPLSCC(0)=0
  DO I=1,NUMNOD
     NTRY=(NSCCRP*I)/NUMNOD
     IRPLSCC(I)=NTRY
     !        IF (MYNOD.EQ.0) WRITE(*,*) "NODE ",I," RPLS ",IRPLSCC(I)
  ENDDO

  ISTART=1+IRPLSCC(MYNOD)
  IFINISH= IRPLSCC(MYNODP)
  !        WRITE(*,*) MYNOD, MYNODP, "ISTART ", 1+IRPLSCC(MYNOD),
  !    $               "IFINISH ", IRPLSCC(MYNODP)

  DO 777 ITER =ISTART,IFINISH

     ! Update coordinates for continuous calculation for SCCDFTB

     DO I = 1,NSCCTC
        C(1,I)=X(IQMLST(I,ITER))/BOHRR
        C(2,I)=Y(IQMLST(I,ITER))/BOHRR
        C(3,I)=Z(IQMLST(I,ITER))/BOHRR
     ENDDO
     DO I = 1,NSCCTC
        EG(1,I)=ZERO
        EG(2,I)=ZERO
        EG(3,I)=ZERO
     ENDDO

     ! Update the MM part as well.

     CALL CHMSCCTB

     IF(NPTC.NE.0) THEN
        DO ICHARM=1,NPTC
           DXTBMM(ICHARM)=ZERO
           DYTBMM(ICHARM)=ZERO
           DZTBMM(ICHARM)=ZERO
        ENDDO
     ENDIF
     ! Clear Mulliken charge 
     IF (LMULIK) THEN
        DO I=1,NSCCTC
           QMULI2(I,ITER)=ZERO
        ENDDO
     ENDIF

     if (period.and.qsccnb) then
        DO ICHARM=1,NPTR
           DXTBM2(ICHARM)=ZERO
           DYTBM2(ICHARM)=ZERO
           DZTBM2(ICHARM)=ZERO
        ENDDO
     endif
     !     QC: UW04_06: in the case of ewald - as far as we do not use
     !     EOPT, then we pre-compute terms related to the MM charges which
     !     do NOT depend on the QM charges 
     NKTMP=1
     if (period.and..not.LSCOEWD) NKTMP=NKVEC

     call chmalloc('sccdftbini.src','SCCTBENE','ISCCEWC',NKTMP,crl=ISCCEWC)
     call chmalloc('sccdftbini.src','SCCTBENE','ISCCEWS',NKTMP,crl=ISCCEWS)
     call chmalloc('sccdftbini.src','SCCTBENE','ISCQEWC',NKTMP,crl=ISCQEWC)
     call chmalloc('sccdftbini.src','SCCTBENE','ISCQEWS',NKTMP,crl=ISCQEWS)

     !     QC: Allocate space for the similar quantities for QM atoms
     !     as well to facilitate derivative calculations involving ewald

     !    XIAO_QC_UW0609: ewald table
     !    PZ MG_QC_UW1206: PME
     if(.not.qsccpme) then
     if (period.and..not.LSCOEWD) &
          call getatomj(kmxscc,isccewc,isccews)
     endif ! qsccpme

     !
     ! allocate space for GHO large arrays ... PJ 7/2004
     !
     IF (QLINK) THEN
        CALL GTDIM(NSCCTC,NGLT,NGSQ)
     ELSE
        NGLT = 1
        NGSQ = 1
     ENDIF
     call chmalloc('sccdftbini.src','SCCTBENE','IPHO',NGLT,crl=IPHO)
     call chmalloc('sccdftbini.src','SCCTBENE','IFAO',NGLT,crl=IFAO)
     call chmalloc('sccdftbini.src','SCCTBENE','ISAO',NGLT,crl=ISAO)
     call chmalloc('sccdftbini.src','SCCTBENE','IWHO',NGLT,crl=IWHO)
     call chmalloc('sccdftbini.src','SCCTBENE','IFHB',NGLT,crl=IFHB)
     call chmalloc('sccdftbini.src','SCCTBENE','ISHB',NGLT,crl=ISHB)
     call chmalloc('sccdftbini.src','SCCTBENE','ICHB',NGSQ,crl=ICHB)
     call chmalloc('sccdftbini.src','SCCTBENE','ICAO',NGSQ,crl=ICAO)

     ! update the hybridization matrix for GHO ... PJ 7/2004
     !
     IF(QLINK) CALL HBDEF(X,Y,Z,BT,BTM,DBTMMM,NQMLNK, &
          IQLINK,JQLINK,KQLINK)

     !     CALL eglcao(nscctc*3,telec,c,scftol,.false.,e,eel,eg, &
     !                      mxitscf,lprvec,qmat,lgrad, &
     CALL eglcao(nscctc*3,      c,       .false.,e,eel,eg, &
          lgrad, &
          isccewc,isccews,iscqewc,iscqews, &
          ewldt,ewrdel,ewnpts, &
          ! GHO ... PJ 7/2004
          ipho,ifao,isao,iwho, &
          ifhb,ishb,ichb,icao)   


     call chmdealloc('sccdftbini.src','SCCTBENE','ISCCEWC',NKTMP,crl =ISCCEWC)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCCEWS',NKTMP,crl =ISCCEWS)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCQEWC',NKTMP,crl =ISCQEWC)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISCQEWS',NKTMP,crl =ISCQEWS)

     !     QC_UW0405 - can we at this stage send the result back to the
     !     Master node - to update info (QMULI2, DX, CTOT) for all replica
     !     computed in this cycle  @@@@
     !     But first see if we can get correct result?

     !     -----------------------------------------------------------
     !     QC_UW04: Add muliken if asked, we transfer over to CG in
     !     energy.src after the nonbond terms are calculated.
     IF (LMULIK) THEN
        DO I=1,NSCCTC
           QMULI2(I,ITER)=QMULIK(I)
        ENDDO
     ENDIF
     !     -----------------------------------------------------------
     CTOT=CTOT + E*TOKCAL*KSCPINT
     !     write(*,*) "calculations done. Node ",MYNOD," Energy ",E*TOKCAL

     DO I=1,NSCCTC
        DX(IQMLST(I,ITER))=DX(IQMLST(I,ITER)) &
             +scal*EG(1,I)*TOKCAL/BOHRR*KSCPINT
        DY(IQMLST(I,ITER))=DY(IQMLST(I,ITER)) &
             +scal*EG(2,I)*TOKCAL/BOHRR*KSCPINT
        DZ(IQMLST(I,ITER))=DZ(IQMLST(I,ITER)) &
             +scal*EG(3,I)*TOKCAL/BOHRR*KSCPINT

        IF(PRNLEV.GT.6) THEN
           WRITE(OUTU,'(I5,3F14.6)') I,EG(1,I)*TOKCAL/BOHRR     &
                , EG(2,I)*TOKCAL/BOHRR &
                , EG(3,I)*TOKCAL/BOHRR
        ENDIF
     ENDDO
     !
     !     MM atoms, without igmsel(i)=-1 !!
     !     QC: CHANGE TO ACCOUT FOR RCUT
     !     QC: YET WE SHOULD ZERO OUT THE QM/MM IMAGE FORCE ON THE IMAGE
     !     ATOM IF DOING INFINITELY DILUTE SYSTEM

     DO I=1,NTOT
        DX(IMMLST(I))=DX(IMMLST(I))+scal*DXTBMM(I)*TOKCAL/BOHRR*KSCPINT
        DY(IMMLST(I))=DY(IMMLST(I))+scal*DYTBMM(I)*TOKCAL/BOHRR*KSCPINT
        DZ(IMMLST(I))=DZ(IMMLST(I))+scal*DZTBMM(I)*TOKCAL/BOHRR*KSCPINT
     ENDDO
     !     QC: Need to add the real space part for eWald - MM
     if (period.and.qsccnb) then
        DO I=1,NPTR
           DX(IMM2LST(I))=DX(IMM2LST(I))+scal*DXTBM2(I)*TOKCAL/BOHRR*KSCPINT
           DY(IMM2LST(I))=DY(IMM2LST(I))+scal*DYTBM2(I)*TOKCAL/BOHRR*KSCPINT
           DZ(IMM2LST(I))=DZ(IMM2LST(I))+scal*DZTBM2(I)*TOKCAL/BOHRR*KSCPINT
        ENDDO
     endif
     !
     ! Determine the GHO Q-link atom derivatives ... PJ 7/2004
     !
     IF(QLINK) THEN
        CALL DQLINK(DX,DY,DZ,X,Y,Z,IPHO,IFAO)
        CALL DQLINK(DX,DY,DZ,X,Y,Z,IWHO,ISAO)
        ECLASS = 0.0D0
        CALL RPMMM(DX,DY,DZ,X,Y,Z,ECLASS)
        CTOT = CTOT + ECLASS
     ENDIF
     !
     ! free space required for GHO gradient calculations ... PJ 7/2004
     !
     call chmdealloc('sccdftbini.src','SCCTBENE','IPHO',NGLT,crl=IPHO)
     call chmdealloc('sccdftbini.src','SCCTBENE','IFAO',NGLT,crl=IFAO)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISAO',NGLT,crl=ISAO)
     call chmdealloc('sccdftbini.src','SCCTBENE','IWHO',NGLT,crl=IWHO)
     call chmdealloc('sccdftbini.src','SCCTBENE','IFHB',NGLT,crl=IFHB)
     call chmdealloc('sccdftbini.src','SCCTBENE','ISHB',NGLT,crl=ISHB)
     call chmdealloc('sccdftbini.src','SCCTBENE','ICHB',NGSQ,crl=ICHB)
     call chmdealloc('sccdftbini.src','SCCTBENE','ICAO',NGSQ,crl=ICAO)

     !     ^^^^^ Iteration end for replica
     !  QC: UW0110: Moved GHO related stuff inside the SCCRP Loop - CHECK PLEASE@@
777 ENDDO

#endif 
  ! ^^^^ if parallel
  !     ----------------------------------------------------------------

#if KEY_PARASCC==1
  IF(MYNOD.EQ.0)THEN
#endif 

     !     QC: Remove codes related to path-integrals

     IF (LSCCRP.AND.LCNSCD) CALL UPCMDXYZ(X,Y,Z,PRNLEV,OUTU)
     IF (LCNSCD) CALL UMBRESCC(X,Y,Z,CTOT,DX,DY,DZ,PRNLEV,OUTU)

#if KEY_PARASCC==1
  ENDIF
#endif 

  RETURN
END SUBROUTINE SCCTBENE

SUBROUTINE SCCTBSEL(ISLCT,QQINP)
  !-----------------------------------------------------------------------
  !     Copies selection vector to common block 
  !     so it may be used by GAMESS interface
  !     Call this routine only once and retain definition
  !     of QM, MM, and link atoms throughout the claculation.
  !     We call this from GAMINI which is called from charmm/charmm.src
  !
  !     IGMSEL(I) = 2  Link atom
  !     IGMSEL(I) = 1  QM atom
  !     IGMSEL(I) = 0  MM atom
  !was:     IGMSEL(I) = -1 MM atom to be excluded from QM/MM interaction
  !     IGMSEL(I) = 5 MM atom to be excluded from QM/MM interaction
  !
  !     MM atom in position close to link atom is excluded from interaction
  !     of external charges to QM region. Instead of this atom is already
  !     a link atom so no need for two atoms in one place!
  !    
  !     QC: All possible replica are considered here.
  !
  use exfunc
  use dimens_fcm
  !
  use coord
  use gamess_fcm
  use stream
  use psf
  use number
  !
  ! added for GHO boundary ... PJ 7/2004
  use stbgho
  use chutil,only:getres,atomid
  !
  implicit none
  !
  INTEGER ISLCT(*)
  LOGICAL QQINP

  INTEGER I,J,I1,I2,N,LN,IS,IQ
  CHARACTER(len=8) SID, RID, REN, AC
  LOGICAL LNFLAG

  !     INTEGER GETRES
  !     EXTERNAL GETRES
  !
  ! MG_QC_UW1206: (bugfix) initialize to zero, otherwise errors possible
  DO I=1, NATOM
    IGMSEL(I)=0
  ENDDO
  DO I=1, NATOM
     !        IGMSEL(I)=ISLCT(I)
     IF (ISLCT(I).EQ.1) THEN
        IGMSEL(I)=1
        IF (ATYPE(I)(1:2) == 'QQ') IGMSEL(I)=2
     ENDIF
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
  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,118)
     WRITE(OUTU,120)  &
          'Classical atoms excluded from the QM calculation'
  ENDIF
118 FORMAT('------------------------------------------------')
120 FORMAT('SCCINT: ',A,':')
122 FORMAT(10X,I5,4(1X,A))
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
  IF(PRNLEV.GE.2) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     !  print out GHO boundary atom ... PJ 7/2004
     WRITE(OUTU, 120) 'Quantum mechanical GHO boundary atoms'
  END IF
  DO I = 1, NQMLNK
     J = IQLINK(I)
     CALL ATOMID(J, SID, RID, REN, AC)
     IF (PRNLEV.GE.2) WRITE(OUTU, 122) J, SID, RID, REN, AC
  ENDDO
  IF(PRNLEV.GE.2) THEN
     IF(NQMLNK .EQ. 0) WRITE (OUTU, 124)
     WRITE(OUTU,118)
  ENDIF
  !
  !     Alow for partial charges on any QM atom
  !
  N=0
  LN=0
  DO I = 1,NATOM
     IF ((IGMSEL(I) .EQ. 1).OR.(IGMSEL(I).EQ.2)) THEN
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
           FQQCHG(N)=-THOSND
        ENDIF
        !
        !          IF(PRNLEV.GE.2) WRITE(OUTU,'(A,2I5,A,F15.5)')
        !    $          'SCCINT: ATOM(',I,N,') has QNUC: ',FQQCHG(N)
     ENDIF
  ENDDO
  !
  ! Zero charges on quantum atoms to remove from MM term.
  IF(QGMREM) THEN
     DO I=1, NATOM
        IF((IGMSEL(I).GT.0).AND.(IGMSEL(I).LT.3)) CG(I)=ZERO
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SCCTBSEL

SUBROUTINE MKMMLST 
  !-----------------------------------------------------------------------
  !     Make a qm/mm list to see who can see QM atoms 
  !     Nothing fancy because we donot have to keep track of which QM
  !     interacts with which MM. 
  !     Use Atom based stuff -- poor choice in most cases
  !     Should be called in the initial set-up procedure and also
  !     whenever image list gets updated.
  !     Potential switching or shift were implemented by scaling the MM
  !     charges to avoid complications with the QM integrals -- should be
  !     small anyhow.
  !
  !     Q. Cui, March 2000.
  !
  use dimens_fcm
  use exfunc
  use blockscc_fcm
  !
  use coord
  use psf   ! iac is in here
  use gamess_fcm
  use sccdftb
  use sccdftbsrc,only: qsccnb,sccfnb,qsccs,qsccsh,lcdko,mmuhub,nmmtype,&
                       mmname,mmuh !MG_QC_UW1206  for KO
  use stream
  use image
  use inbnd
  use parallel
  use chutil,only:getres
  use rtf,only:atct    !MG_QC_UW1206  for KO
  implicit none

  INTEGER I,I1,I2,J,N
  real(chm_real) RQMMM2,R2CUTQM
  INTEGER NPRIMM,NTRY

  !     INTEGER GETRES
  !     EXTERNAL GETRES

  logical cshft,cshift
  ! XIAO_QC_UW0609 DIV:
  INTEGER K, L, IS, IQ
  REAL(chm_real) DQ, NQ, NE
  character(len=6) ele !MG_QC_UW1206  for KO

  !     QC: Restructure the code (Thu May 13 17:41:24 CDT 2004)
  !     Several possibilities:
  !     1. No cutoff (no PBC) --------> ALL primary MM atoms are packed
  !     2. QM/MM Ewald w/o qsccnb  ---> ALL primary MM atoms
  !     3. qsccnb but no Ewald -------> Both primary and image within cut
  !     4. qsccnb w/ Ewald  ----------> Pack ALL primary MM into IMMLST
  !                                     for reciprocal
  !                                     Also need image MM atoms into 
  !                                     another list - to minimize the
  !                                     amount of time for the real space
  !                                     sum
  if ((.not.qsccnb).or.period) then
     !     .........  Case 1 & 2,4 ..........
     NPTC=0
     DO J=1,NATOM
        IF(IGMSEL(J).EQ.0) THEN
           NPTC = NPTC + 1
           IMMLST(NPTC) = J
           ZPTC(NPTC)=CG(J)

! MG+Guanhua_QC_UW1206: find out the element type for mm atoms
           if (lcdko) then
!       call findel(atct(iac(j)),amass(j),nptc,ele,aznuc(j),'f')
           ele=atct(iac(j))
           mmuhub(nptc)=0.0d0
           do i=1, nmmtype
             if (ele(1:1) .eq. mmname(i)) then
               mmuhub(nptc)=mmuh(i)
             endif
           enddo
           endif

           ! XIAO_QC_UW0609 DIV
           IF (QSCCDIV) THEN
              ! check if within group of atom J there is an atom
              ! whose charge should be distributed
              K=GETRES(J,IGPBS,NGRP)
              IS=IGPBS(K)+1
              IQ=IGPBS(K+1)
              DQ=0
              NQ=0
              NE=0
              DO L=IS,IQ
                 IF (IGMSEL(L).EQ.5) THEN
                    DQ=DQ+CG(L)
                    NE=NE+1
                 ENDIF
                 IF (IGMSEL(L).EQ.0) THEN
                    NQ=NQ+1
                 ENDIF
              ENDDO
              IF (NE.gt.0) THEN
                 ZPTC(NPTC)=ZPTC(NPTC)+DQ/NQ
              ENDIF
           ENDIF
        ENDIF
     ENDDO
#if KEY_PARALLEL==1
     !       CREAT A LIST TO RECORD BOUNDARY MM atoms in different nodes
     !     For parallel computations - distribute the kvectors to different
     !     NODES
     IMMLSC(0)=0
     DO I=1,NUMNOD
        NTRY=(NPTC*I)/NUMNOD
        IMMLSC(I)=NTRY
        !        IF (MYNOD.EQ.0) WRITE(*,*) "NODE ",I," MMLS ",IMMLSC(I)
     ENDDO
#endif 

     IF (.not.(qsccnb.and.period)) RETURN
  endif

  !     .............. Case 3,4 ...................
  !     if(qsccnb) then
  CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT
  CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT
  rcutqm=cutnb
  qsccs=cshft
  qsccsh=cshift
  sccfnb=ctofnb
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,'(1x,"MMLST> Update QM/MM List with RCUT:",2F10.5)') &
          RCUTQM,sccfnb
     WRITE(OUTU,*) "MMLST> Other parameters: ",qsccs,qsccsh
#if KEY_PARALLEL==1
  ENDIF
#endif 
  !     endif

  if (period) go to 700

  !     Case 3
  ! ------------------------------------------------------------------
  NPTC=0
  R2CUTQM=RCUTQM*RCUTQM
  DO 100 J = 1,NATOM
     IF(IGMSEL(J).EQ.0) THEN
        !         DO 200 I = 1,NATOM
        !            IF(IGMSEL(I).GE.1)THEN
        DO I1=1,NSCCRP
           DO I2=1,NSCCTC
              I=IQMLST(I2,I1)
              RQMMM2 = (X(J) - X(I))*(X(J)-X(I))  &
                   + (Y(J) - Y(I))*(Y(J)-Y(I))  &
                   + (Z(J) - Z(I))*(Z(J)-Z(I)) 
              IF (RQMMM2.LE.R2CUTQM) THEN 
                 NPTC = NPTC + 1
                 IMMLST(NPTC) = J
                 ZPTC(NPTC)=CG(J)

!!Puja_QC_UW0113
                 IF (QSCCDIV) THEN
                 ! check if within group of atom J there is an atom
                 ! whose charge should be distributed
                   K=GETRES(J,IGPBS,NGRP)
                   IS=IGPBS(K)+1
                   IQ=IGPBS(K+1)
                   DQ=0
                   NQ=0
                   NE=0
                   DO L=IS,IQ
                     IF (IGMSEL(L).EQ.5) THEN
                       DQ=DQ+CG(L)
                       NE=NE+1
                     ENDIF
                     IF (IGMSEL(L).EQ.0) THEN
                       NQ=NQ+1
                     ENDIF
                   ENDDO
                   IF (NE.gt.0) THEN
                     ZPTC(NPTC)=ZPTC(NPTC)+DQ/NQ
                   ENDIF
                 ENDIF
!!

                 GOTO 210
              ENDIF
           ENDDO
        ENDDO
        !            ENDIF
        !200      ENDDO
210     CONTINUE 
     ENDIF
100 ENDDO

#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,*) " MMLST> PRIMARY MM: NPTC= ",NPTC
#if KEY_PARALLEL==1
  ENDIF
#endif 
  NPRIMM = NPTC

  !     QC: ALSO ADD IN THE IMAGE ATOMS IF DEFINED 
  IF ((NATIM.EQ.0).OR.LSKIMG) RETURN
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,*) " MMLST> PACK IMAGE"
#if KEY_PARALLEL==1
  ENDIF
#endif 
  DO 300 J = NATOM + 1, NATIM
     IF(IGMSEL(J).EQ.0) THEN
        !         DO 400 I = 1,NATOM
        !            IF(IGMSEL(I).GE.1)THEN
        DO I1=1,NSCCRP
           DO I2=1,NSCCTC
              I=IQMLST(I2,I1)  
              RQMMM2 = (X(J) - X(I))*(X(J)-X(I))  &
                   + (Y(J) - Y(I))*(Y(J)-Y(I))  &
                   + (Z(J) - Z(I))*(Z(J)-Z(I)) 
              IF (RQMMM2.LE.R2CUTQM) THEN 
                 NPTC = NPTC + 1
                 IMMLST(NPTC) = J
                 ZPTC(NPTC)=CG(J)
                 GOTO 410
              ENDIF
           ENDDO
        ENDDO
        !            ENDIF
        !400      ENDDO
410     CONTINUE 
     ENDIF
300 ENDDO
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,*) " MMLST> MM IMAGE ADDED:  NPTC= ",NPTC-NPRIMM
#if KEY_PARALLEL==1
  ENDIF
#endif 
  RETURN
  ! ------------------------------------------------------------------
700 CONTINUE ! Case 4 for packing another set of arrays for real
  !                space sum

  NPTR=0
  R2CUTQM=RCUTQM*RCUTQM
  DO 800 J = 1,NATOM
     IF(IGMSEL(J).EQ.0) THEN
        !         DO 900 I = 1,NATOM
        !            IF(IGMSEL(I).GE.1)THEN
        DO I1=1,NSCCRP
           DO I2=1,NSCCTC
              I=IQMLST(I2,I1)
              RQMMM2 = (X(J) - X(I))*(X(J)-X(I))  &
                   + (Y(J) - Y(I))*(Y(J)-Y(I))  &
                   + (Z(J) - Z(I))*(Z(J)-Z(I)) 
              IF (RQMMM2.LE.R2CUTQM) THEN 
                 NPTR = NPTR + 1
                 IMM2LST(NPTR) = J
                 ZPTR(NPTR)=CG(J)
!!Puja_QC_UW0113
                 IF (QSCCDIV) THEN
                 ! check if within group of atom J there is an atom
                 ! whose charge should be distributed
                     K=GETRES(J,IGPBS,NGRP)
                     IS=IGPBS(K)+1
                     IQ=IGPBS(K+1)
                     DQ=0
                     NQ=0
                     NE=0
                     DO L=IS,IQ
                        IF (IGMSEL(L).EQ.5) THEN
                           DQ=DQ+CG(L)
                           NE=NE+1
                        ENDIF
                        IF (IGMSEL(L).EQ.0) THEN
                           NQ=NQ+1
                        ENDIF
                     ENDDO
                     IF (NE.gt.0) THEN
                        ZPTR(NPTR)=ZPTR(NPTR)+DQ/NQ
                     ENDIF
                 ENDIF
!!
                 GOTO 910
              ENDIF
           ENDDO
        ENDDO
        !            ENDIF
        !900      ENDDO

910     CONTINUE 
     ENDIF
800 ENDDO

#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,*) "MM2LST> PRIMARY MM: NPTC= ",NPTR
#if KEY_PARALLEL==1
  ENDIF
#endif 
  NPRIMM = NPTR

  !     QC: ALSO ADD IN THE IMAGE ATOMS IF DEFINED 
  IF ((NATIM.EQ.0).OR.LSKIMG) RETURN
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,*) "MM2LST> PACK IMAGE"
#if KEY_PARALLEL==1
  ENDIF
#endif 
  DO 980 J = NATOM + 1, NATIM
     IF(IGMSEL(J).EQ.0) THEN
        !         DO 990 I = 1,NATOM
        !            IF(IGMSEL(I).GE.1)THEN
        DO I1=1,NSCCRP
           DO I2=1,NSCCTC   
              I=IQMLST(I2,I1) 
              RQMMM2 = (X(J) - X(I))*(X(J)-X(I))  &
                   + (Y(J) - Y(I))*(Y(J)-Y(I))  &
                   + (Z(J) - Z(I))*(Z(J)-Z(I)) 
              IF (RQMMM2.LE.R2CUTQM) THEN 
                 NPTR = NPTR + 1
                 IMM2LST(NPTR) = J
                 ZPTR(NPTR)=CG(J)
                 GOTO 991
              ENDIF
              !            ENDIF
           ENDDO
        ENDDO
        !990      ENDDO
991     CONTINUE 
     ENDIF
980 ENDDO
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     WRITE(OUTU,*) "MM2LST> MM IMAGE ADDED:  NPTC= ",NPTR-NPRIMM
#if KEY_PARALLEL==1
  ENDIF
#endif 

  RETURN
END SUBROUTINE MKMMLST

#if KEY_PBEQ==1
SUBROUTINE UPSCCMM(NTRB,LSTRB)
  !-----------------------------------------------------------------------
  !     Make case 5: GSBP (NTRB.ne.0) ->Consider ONLY the inner atoms
  !                                     in MM list to save time
  !     QC: Wed Jun  9 17:52:30 CDT 2004
  !
  use dimens_fcm
  use psf
  use gamess_fcm
  use sccdftb
  use sccgsbp
  use stream
  ! XIAO_QC_UW0609 add for div
  use exfunc
  use chutil,only:getres

  implicit none

  INTEGER NTRB,LSTRB(*)

  INTEGER I,J,N

  !     INTEGER GETRES
  !     EXTERNAL GETRES

  ! XIAO_QC_UW0609 DIV:
  INTEGER K, L, IS, IQ
  REAL(chm_real) DQ, NQ, NE

  IF (NTRB.EQ.0) THEN
     CALL WRNDIE(0,'<UPSCCMM>','NO INNER ATOMS TO PACK.')
     RETURN
  ENDIF

  WRITE(OUTU,*) "REPACKING GSBP INNER ATOMS out of: ",NTRB
  WRITE(OUTU,*) "ORIGINAL MM: ",NPTC
  NPTC=0
  DO I=1,NTRB
     J=LSTRB(I) 
     IF(IGMSEL(J).EQ.0) THEN
        NPTC = NPTC + 1
        IMMLST(NPTC) = J
        ZPTC(NPTC)=CG(J)
        ! added by XIAO_QC_UW0609 (DIV)
        ! XIAO_QC_UW0609 DIV
        ! slightly complicated because no reverse map exists for
        ! IMMLST
        IF (QSCCDIV) THEN
           ! check if within group of atom J there is an atom
           ! whose charge should be distributed
           K=GETRES(J,IGPBS,NGRP)
           IS=IGPBS(K)+1
           IQ=IGPBS(K+1)
           DQ=0
           NQ=0
           NE=0
           DO L=IS,IQ
              IF (IGMSEL(L).EQ.5) THEN
                 DQ=DQ+CG(L)
                 NE=NE+1
              ENDIF
              IF (IGMSEL(L).EQ.0) THEN
                 NQ=NQ+1
              ENDIF
           ENDDO
           IF (NE.gt.0) THEN
              ZPTC(NPTC)=ZPTC(NPTC)+DQ/NQ
           ENDIF
        ENDIF
     ENDIF
  ENDDO
  WRITE(OUTU,*) "REPACKED MM: ",NPTC

  !     WE DO THIS UPDATE ONLY ONCE
  LSCCMM=.TRUE.

  RETURN
END SUBROUTINE UPSCCMM

SUBROUTINE UPSCCSURF(NUMSURF,SURFCRD,SURFCHR)
!-----------------------------------------------------------------------
!     Make case 6: SMBP (NSURF.ne.0) -> Add Surface Charges to
!                                       list of External Charges
!     JZ_UW12 from UPSCCMM
!
  use dimens_fcm
  use psf
  use gamess_fcm
  use sccdftb
  use sccdftbsrc
  use sccgsbp
  use stream
  implicit none
  real(chm_real) SURFCRD(*),SURFCHR(*)
  INTEGER NUMSURF
  ! local      
  INTEGER I,J,N

  ! JZ_UW12: SMBP currently only implemented for 'No cutoff' (case 1 below)
      IF (qsccnb .OR. period) THEN
        CALL WRNDIE(-5,'<UPSCCSURF>','SMBP NYI for CUTF/EWAD')
      ENDIF

      IF (NUMSURF.EQ.0) THEN
        CALL WRNDIE(0,'<UPSCCSURF>','NO SURFACE CHARGES PRESENT.')
        RETURN
      ENDIF

      IF (PRNLEV .ge. 5) THEN
        WRITE(OUTU,102) "Adding following number of surface charges: ", NUMSURF
        WRITE(OUTU,102) "Number of MM atoms:                         ", NPTC
      ENDIF
      J = NATOM

      DO I = 1, NUMSURF
        NPTC = NPTC + 1
        J    = J    + 1
        IMMLST(NPTC) = J
        ZPTC(NPTC)   = SURFCHR(I)
      ENDDO
      WRITE(OUTU,102) "Total number of external point charges:     ", NPTC

 102  FORMAT(3X,A,I10)
      RETURN
      END SUBROUTINE UPSCCSURF
#endif /* PBEQ*/

SUBROUTINE updbox(boxsiz,XTLABC,IFLAG,NLAT)
  use stream
  use consta
  use sccdftb, only: maxkscc,kxvec,kyvec,kzvec,kvec,nkvec,kxv,kyv,kzv, &
       kappascc,kmxscc,ksqmxscc,LSCOEWD
  use sccdftbsrc

  implicit none

  real(chm_real) boxsiz(3,3)
  real(chm_real) XTLABC(6)
  INTEGER NLAT(3)
  INTEGER IFLAG
  !     INTEGER NTYPE
  !     common /sccntype/ ntype

  !     integer maxtyp
  !     parameter( maxtyp= 6)
  !     integer maxtab
  !     parameter( maxtab= 600)

  !     real(chm_real) skhtab(10,maxtab,maxtyp,maxtyp),  &
  !            skstab(10,maxtab,maxtyp,maxtyp)  
  !     real(chm_real) skself(3,maxtyp), sr(maxtyp,maxtyp)
  !     INTEGER dim(maxtyp,maxtyp)

  !     common /sktab/ skhtab,skstab,skself,sr,dim

  !     If choose to optimize para
  !     logical LSCOEWD 
  !     real(chm_real) kappascc
  !     integer kmxscc,ksqmxscc
  !     common /sccewd/ kappascc,kmxscc,ksqmxscc,LSCOEWD
  !     integer maxkscc
  !     parameter(maxkscc=5000)
  !     real(chm_real) kxvec(maxkscc)
  !     real(chm_real) kyvec(maxkscc)
  !     real(chm_real) kzvec(maxkscc)
  !     real(chm_real) kvec (maxkscc)
  !     integer nkvec
  !     common /sccewk/ kxvec,kyvec,kzvec,kvec,nkvec
  !     XIAO_QC_UW0609: add new vector for cos and sin calculation
  !      integer kxv(maxkscc)
  !      integer kyv(maxkscc)
  !      integer kzv(maxkscc)
  !      /MH09/ already in the module
  !      common /sccewk2/ kxv,kyv,kzv


  !     LOCAL VARIABLE
  INTEGER  I,J
  real(chm_real) yhlp,slkcutoff
  real(chm_real) recbasis(3,3), vol
  real(chm_real) alpha
  real(chm_real) getalpha
  EXTERNAL getalpha


  !     Transfer unit cell info from crystal: basis and 
  !     size (note in cpt, the basis and size might change)

  WRITE(OUTU,*) "Symmetric unit cell parameters:"
  IF(PRNLEV.GE.2) THEN
     WRITE (OUTU,'(A)') ' Lattice Vectors :'
     WRITE (OUTU,'(A,3F16.10)') ' A = ',XTLABC(1),XTLABC(2),XTLABC(4)
     WRITE (OUTU,'(A,3F16.10)') ' B = ',XTLABC(2),XTLABC(3),XTLABC(5)
     WRITE (OUTU,'(A,3F16.10)') ' C = ',XTLABC(4),XTLABC(5),XTLABC(6)
  ENDIF

  !       convert to boxsiz as required by SCC - do NOT forget about
  !       unit conversion

  boxsiz(1,1)=XTLABC(1)/BOHRR
  boxsiz(1,2)=XTLABC(2)/BOHRR
  boxsiz(1,3)=XTLABC(4)/BOHRR
  boxsiz(2,1)=XTLABC(2)/BOHRR
  boxsiz(2,2)=XTLABC(3)/BOHRR
  boxsiz(2,3)=XTLABC(5)/BOHRR
  boxsiz(3,1)=XTLABC(4)/BOHRR
  boxsiz(3,2)=XTLABC(5)/BOHRR
  boxsiz(3,3)=XTLABC(6)/BOHRR

  !       Set up/update the K-space table
  !       get reciprocal lattice vectors and cell volume
  CALL REZVOL(boxsiz,recbasis,vol)
  alpha = getalpha(boxsiz)


  write(outu,*) "Setting up k-vectors for SCC reciprocal sum"
  !       XIAO_QC_UW0609: add kxv,kyv,kzv
  call ktabscc(kxvec,kyvec,kzvec,kvec,recbasis,alpha,vol, &
       nkvec,kmxscc,ksqmxscc,maxkscc,kxv,kyv,kzv)

  !       Update nlat if necessary

  IF (IFLAG.NE.1) RETURN
  slkcutoff=0.0d0
  do i = 1,ntype 
     do j = 1,ntype 
        yhlp=sr(i,j)*dim(i,j)+0.3
        if(yhlp.gt.slkcutoff)then
           slkcutoff=yhlp
        endif
     enddo
  enddo
  write(outu,*) "updbox: slkcutoff>",slkcutoff,"ntype>",ntype
  call gamma_summind(slkcutoff,boxsiz,nlat)
  write(outu,*) 'Number of lattice sums for Gammapoint matrix:',  &
       nlat(1), nlat(2), nlat(3) 

  RETURN
END SUBROUTINE updbox
!     Haibo March 2 2005 
SUBROUTINE UPCMDXYZ(X,Y,Z,PRNLEV,OUTU)

  use number
  use sccdftb
  implicit none

  real(chm_real) X(*),Y(*),Z(*)
  INTEGER PRNLEV,OUTU

  INTEGER I,J

  DO I = 1, NSCCTC
     XYZCMD(1,I) = ZERO
     XYZCMD(2,I) = ZERO
     XYZCMD(3,I) = ZERO
     DO  J =1,NSCCRP
        XYZCMD(1,I) = XYZCMD(1,I) + X(IQMLST(I,J))
        XYZCMD(2,I) = XYZCMD(2,I) + Y(IQMLST(I,J))
        XYZCMD(3,I) = XYZCMD(3,I) + Z(IQMLST(I,J))
     ENDDO
     XYZCMD( 1,I) = XYZCMD(1,I) / DBLE(NSCCRP)
     XYZCMD( 2,I) = XYZCMD(2,I) / DBLE(NSCCRP)
     XYZCMD( 3,I) = XYZCMD(3,I) / DBLE(NSCCRP)
  ENDDO

  IF (PRNLEV.GE.6) THEN
     WRITE(OUTU,*) "Centroid position updated for ",NSCCTC, &
          " replica"
     DO I=1,NSCCTC
        WRITE(OUTU,'(1x,"CMDXYZ> ",I5, 3F12.5)') I,XYZCMD(1,I), &
             XYZCMD(2,I),XYZCMD(3,I)
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE UPCMDXYZ
!
!     Haibo Yu March 2 2005
SUBROUTINE UMBRESCC(X,Y,Z,CTOT,DX,DY,DZ,PRNLEV,OUTU)
  !-----------------------------------------------------------------------
  !
  !PATINT     The Harmonic constraints on the centroid
  !
  !     Q. Cui, Xmas, 1998.
  !
  use dimens_fcm
  use sccdftb
  use consta
  use gamess_fcm
  use number
  !
  implicit none

  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  real(chm_real) CTOT
  INTEGER PRNLEV,OUTU,KK
  !

  !     LOCAL VARIABLES
  real(chm_real) RTMP1(100),RTMP2(100)
  real(chm_real) XTMP1(100),XTMP2(100)
  real(chm_real) YTMP1(100),YTMP2(100)
  real(chm_real) ZTMP1(100),ZTMP2(100)
  INTEGER I,II
  real(chm_real) EUMBRELLA
  real(chm_real) FXTMP1(100),FYTMP1(100),FZTMP1(100)
  real(chm_real) FXTMP2(100),FYTMP2(100),FZTMP2(100)
  real(chm_real) RTMP0

  !     First all the temp variables

  !     Haibo Yu We loop all the restraining here
  RTMP0 = 0
  DO II = 1, NCNSCD
     IF (PRNLEV.GE.6) WRITE(OUTU,'(1x,"AUM> ",3I10)') &
          IATCNS((II-1)*3+1),IATCNS((II-1)*3+2),IATCNS((II-1)*3+3)
     IF (PRNLEV.GE.6) WRITE(OUTU,'(1x,"XYZ> ",9F10.5)') &
          XYZCMD(1,IATCNS((II-1)*3+1)),XYZCMD(2,IATCNS((II-1)*3+1)), &
          XYZCMD(3,IATCNS((II-1)*3+1)), &
          XYZCMD(1,IATCNS((II-1)*3+2)),XYZCMD(2,IATCNS((II-1)*3+2)), &
          XYZCMD(3,IATCNS((II-1)*3+2)), &
          XYZCMD(1,IATCNS((II-1)*3+3)),XYZCMD(2,IATCNS((II-1)*3+3)), &
          XYZCMD(3,IATCNS((II-1)*3+3))
     XTMP1(II)=XYZCMD(1,IATCNS((II-1)*3+1)) - &
          XYZCMD(1,IATCNS((II-1)*3+2))
     YTMP1(II)=XYZCMD(2,IATCNS((II-1)*3+1)) -  &
          XYZCMD(2,IATCNS((II-1)*3+2))
     ZTMP1(II)=XYZCMD(3,IATCNS((II-1)*3+1)) -  &
          XYZCMD(3,IATCNS((II-1)*3+2))
     RTMP1(II)=dsqrt(XTMP1(II)*XTMP1(II) +  &
          YTMP1(II)*YTMP1(II) + ZTMP1(II)*ZTMP1(II))

     XTMP2(II)=XYZCMD(1,IATCNS((II-1)*3+2)) -  &
          XYZCMD(1,IATCNS((II-1)*3+3))
     YTMP2(II)=XYZCMD(2,IATCNS((II-1)*3+2)) -  &
          XYZCMD(2,IATCNS((II-1)*3+3))
     ZTMP2(II)=XYZCMD(3,IATCNS((II-1)*3+2)) -  &
          XYZCMD(3,IATCNS((II-1)*3+3))
     RTMP2(II)=dsqrt(XTMP2(II)*XTMP2(II) +  &
          YTMP2(II)*YTMP2(II) + ZTMP2(II)*ZTMP2(II))
     RTMP0 = RTMP0 + (RTMP1(II)-RTMP2(II))/DBLE(NCNSCD) 
     !
  ENDDO
  !     Putting forces to the quasiparticles
  DO II = 1, NCNSCD 
     FXTMP1(II)=FCNSCD*(RTMP0 - RCNSCD0)/RTMP1(II)* &
          XTMP1(II)/DBLE(NSCCRP)/DBLE(NCNSCD)
     FYTMP1(II)=FCNSCD*(RTMP0 - RCNSCD0)/RTMP1(II)* &
          YTMP1(II)/DBLE(NSCCRP)/DBLE(NCNSCD)
     FZTMP1(II)=FCNSCD*(RTMP0 - RCNSCD0)/RTMP1(II)* &
          ZTMP1(II)/DBLE(NSCCRP)/DBLE(NCNSCD)
     FXTMP2(II)=FCNSCD*(RTMP0 - RCNSCD0)/RTMP2(II)* &
          XTMP2(II)/DBLE(NSCCRP)/DBLE(NCNSCD)
     FYTMP2(II)=FCNSCD*(RTMP0 - RCNSCD0)/RTMP2(II)* &
          YTMP2(II)/DBLE(NSCCRP)/DBLE(NCNSCD)
     FZTMP2(II)=FCNSCD*(RTMP0 - RCNSCD0)/RTMP2(II)* &
          ZTMP2(II)/DBLE(NSCCRP)/DBLE(NCNSCD)
     DO I=1,NSCCRP
        DX(IQMLST(IATCNS((II-1)*3+1),I))= &
             DX(IQMLST(IATCNS((II-1)*3+1),I))+FXTMP1(II)
        DY(IQMLST(IATCNS((II-1)*3+1),I))= &
             DY(IQMLST(IATCNS((II-1)*3+1),I))+FYTMP1(II)
        DZ(IQMLST(IATCNS((II-1)*3+1),I))= &
             DZ(IQMLST(IATCNS((II-1)*3+1),I))+FZTMP1(II)
        DX(IQMLST(IATCNS((II-1)*3+2),I))= &
             DX(IQMLST(IATCNS((II-1)*3+2),I))-FXTMP1(II)-FXTMP2(II)
        DY(IQMLST(IATCNS((II-1)*3+2),I))= &
             DY(IQMLST(IATCNS((II-1)*3+2),I))-FYTMP1(II)-FYTMP2(II)
        DZ(IQMLST(IATCNS((II-1)*3+2),I))= &
             DZ(IQMLST(IATCNS((II-1)*3+2),I))-FZTMP1(II)-FZTMP2(II)
        DX(IQMLST(IATCNS((II-1)*3+3),I))= &
             DX(IQMLST(IATCNS((II-1)*3+3),I))+FXTMP2(II)
        DY(IQMLST(IATCNS((II-1)*3+3),I))= &
             DY(IQMLST(IATCNS((II-1)*3+3),I))+FYTMP2(II)
        DZ(IQMLST(IATCNS((II-1)*3+3),I))= &
             DZ(IQMLST(IATCNS((II-1)*3+3),I))+FZTMP2(II)
     ENDDO
  ENDDO

  EUMBRELLA=(RTMP0 - RCNSCD0)*0.5d0*FCNSCD* &
       (RTMP0 - RCNSCD0)

  IF (PRNLEV.GE.6) WRITE(OUTU,'(1x,"EUMBR> ",F12.5)') &
       EUMBRELLA
  IF (PRNLEV.GE.6) WRITE(OUTU,'(1x,"DISBR> ",2F10.5)') &
       RTMP0,RCNSCD0

  CTOT=CTOT + EUMBRELLA


  ISCDWRT=ISCDWRT + 1
  JSCDWRT = JSCDWRT + 1
  IF (MOD(ISCDWRT,NFRQSC1).EQ.0) THEN
     R1SCD(JSCDWRT) = RTMP0
  ENDIF

  !     OUTPUT STUFF IF NECC
  IF (MOD(JSCDWRT,NFRQSCW).EQ.0) THEN
     IF (UNTSCD.NE.0) THEN
        WRITE(UNTSCD,'(F15.5)') R1SCD(JSCDWRT)
     ENDIF
     JSCDWRT = 0
  ENDIF
  return
end SUBROUTINE UMBRESCC

subroutine chardkoread
      use sccdftbsrc, only:lcdko,kalpha,kbeta,nmmtype,mmname,mmuh
      integer itype,ntypej,izipj,i
      !common /kopara/ lcdko,kalpha,kbeta,nmmtype,mmname,mmuh
      !save /kopara/

      open(unit=54,file='ko_para.inp',status='unknown')
      rewind (54)
      read (54,*) ntypej,nmmtype
      do itype=1, ntypej
        read (54,*) izipj,kalpha(izipj),kbeta(izipj)
        write(6,*) izipj,kalpha(izipj),kbeta(izipj)
      enddo
      read (54,*)
      ! Guanhua: readin MM Hubbard parameter
      if (nmmtype .ne. 0) then
        do i=1,nmmtype
          read (54,*) mmname(i),mmuh(i)
          mmname(i)=adjustl(mmname(i))
        enddo
      endif

      close(54)

      return
end subroutine chardkoread
#endif 
