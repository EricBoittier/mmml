module galgor

  !-----------------------------------------------------------------------
  !  Developed by Michal Vieth, Heidi Daigler and  Charles Brooks III
  !  at The Scripps Research Institute during the summer/fall of 1997 based
  !  on the code provided by Heidi Daigler and Charles Brooks, Department of
  !  Chemistry, Carnegie Mellon University developed during the summer of 1994.
  !  Its purpose is to enable monte carlo and genetic algorithm based
  !  conformational searches to be performed on peptides/proteins, small
  !  organic molecules and docking of (small) ligands to their receptors.
  !  It builds upon the replica ideas of Leo Caves to make multiple copies
  !  of the system, i.e., the chromosomes.  These chromosomes make up a
  !  population of molecular conformations which are crossed and mutated
  !  according to the specific genetic algorithm used.  It is recomended to
  !  use a generational update version of genetic algorithm with elitism 1 or 2,
  !  and island/nicheing ideas to evolve subpopulations independently.
  !  Each chromosome is represented by the set of internal coordinates
  !  (and rigid body degrees of freedom if desired) deemed to "evolve", i.e.,
  !  the genes.  The genes are represented as real numbers.  The chromosome
  !  are crossed at a randomly chosen gene or random genes are mutated with
  !  a given rate of mutation.  In addition migration of individuals
  !  between subpopulations is used if the island model is employed.
  !  From each population, the members suitable for crossing, the "parents",
  !  are chosen based upon their ranking, as measured by a priori chosen
  !  preference to select higher ranked individuals. Evolutionary presure of
  !  1.2 based of scaled fitness following Jones, JMB 245, 43-53 is used for
  !  parents selection.
  !  The potential energy function used can be all of CHARMM's energy
  !  terms, including constraints, or any subset of them.  The user energy
  !  term can be utilized to include any special functional form which is
  !  particularly amenable to the GA search.  The current implementation
  !  uses the standard CHARMM energy functions.  It is possible that a
  !  special "bump" energy function will be implemented to permit fast
  !  evaluation of the non-bonded overlaps in any structure (chromosome).
  !
  !  The basic code is implemented following the outline:
  !
  !  1.  Parse the control variables based upon the keyword
  !      GALGorithm.  These include:
  !
  !    GALGorithm SETUp             !replicate subsystem and set-up data
  !                                 !structures
  !      CHROmosomes <int>  <atom selection>
  !                                 !number of chromosomes to replicate
  !                                 !
  !      {SEED 3x(<resnum> <atom>)} ! Specify seed atoms for ic builds
  !        DIHEdral {DEPE} {IMPRO} {ALL} {NONE} {INCLude} {EXCLude}
  !        4x<atom selection> <int>
  !                                 ! (IMPR) fpllowed by an atom selection
  !                                 !indicated the gene with impoper dihedral
  !        ANGLe {ALL} {INCLude} {EXCLude} 3x<atom selection>
  !        BOND {ALL} {INCLude} {EXCLude} 2x<atom selection>
  !        TRAN ROTA                !presence of this key words turns on
  !                                 !6 genes with rigid body degrees of freedom
  !      END                        !termination for VARI sub-parser
  !                                 !the default is that all dihedral ICs
  !                                 !will be mobile and no angles or
  !                                 !bonds.  However, one can INCLude
  !                                 !or EXCLude any or all of any IC defined
  !                                 !in the IC table originally set-up.
  !      GALGorithm EVOLve          !execute the GA or MC evolution
  !      MAXGenerations <int>       !maximum number of generations
  !      TOLErance <real>           !energy tolerance for the convergence
  !                                 !of the population
  !      TOLCoordinate <real>       !coordinate tolerance for the convergence
  !                                 !of the population
  !      PRINt_frequency <int>      !frequency for printing energies of
  !                                 !individual chromosomes
  !      RANDomize                  !randomize selected ICs to initialize
  !                                 !GA search.
  !      MUTAtion_rate <real>
  !                                 !rate of random mutations in random genes
  !                                 !is given by MUTA.  The maximum size of
  !                                 !any mutation is specific for the IC type
  !                                 !(see DIHE, ANGL or BOND below) but is
  !                                 !altered by the profile determined from
  !                                 !ON, OFF, HEIGht and SLOPe.  It is
  !                                 !held constant for ON generations and
  !                                 !then reduced to a lower plateau by OFF
  !                                 !at which point its held constant.  HEIGht
  !                                 !and SLOPe determine the reduction.
  !      PAREnts <int>              !number of parents to be used in crossing
  !      GSIZe <int>                !number of "bits" into which a gene is
  !                                 !divided, the defaults is 1 and is strongly
  !                                 !recomdended
  !      CHILdren <int>             !the number of children to be generated
  !                                 !by mating
  !      ELITE    <int>             !the elitism of the population, the numnber
  !                                 !of most fit parents to be left in the
  !                                 !new population. It is active only
  !                                 !if generational update is used, default
  !                                 !recomended value is 2
  !      NICHes <int> INTEraction <int>
  !                                 !number of niches to propagate and how
  !                                 !frequently they interact through crossing
  !      DISTep <real> ANSTep <real> BOSTep <real)
  !                    TRANstep <real> ROTAstep <real>
  !                                 !specify the step sizes for mutations in
  !                                 !dihedrals (DISTep), angles (ANSTep) and
  !                                 !bonds (BOSTep), translations (TRANstep)
  !                                 !and rotations in radians (ROTAstep)
  !
  !      NPRInt <int>               !the number of most fit chromosomes
  !                                 !whose energies will be printed
  !      PALL <int>                 !relative probability to mutate all 3
  !                                 !translational or rotational degrees of
  !                                 ! freedom, default 0 with respect to other
  !                                 ! rigid body mutations, default 0
  !      PCALl <int>                !relative probability to mutate all 6 rigid
  !                                 ! body degrees of freedom, default 0
  !      PTALl <int>                !probability to mutate all interanl degrees
  !                                 !of freedom simultaneously, default 0
  !      PINTernal                  !probability to mutate
  !                                 ! internal degrees of
  !                                 !freedom, the probability to mutate rigid
  !                                 !body degrees of freedom is 1-pint
  !
  !      CLEAr                      !clear GA data structures (not replica)
  !
  !
  !  In the initial set-up stage, the GALGorithm keyword is parsed and the
  !  main subroutine for the genetic algorithm is called from CHARMM
  !  (GENETIC).  The keyword SETUp causes a number of things to occur.
  !  First the number of chromosomes (replicas) and the atoms whose
  !  internal coordinates are to be sampled and replicated are parsed.
  !  Following this, the specific IC variables which are to be "evolved"
  !  are selected by the VARIable_ICs sub-parser.  This is carried out in a
  !  separate routine which is modeled after the routine used for
  !  processing the IC edit command.  Finally, the replica subroutine from
  !  Leo Caves is called to replicat the system and the original subsystem
  !  atoms are deleted by a call equialent to delete atoms select segid
  !  orig end.  Each new segment generated by the replica command is given
  !  the prefix C (for chromosome) followed by the number of that replica
  !  (chromosome).  The approopriate pointer structures for
  !  inclusion/exclusion of dihedral, angle and bond ICs are then created
  !  and control is passed back to the CHARMM level, where specific
  !  manipulations can be performed on the individual chromosomes, e.g., to
  !  vary initial conformations around the initial progenerator of all
  !  chromosomes.
  !
  !  Evolution of the population of chromosomes is controlled by calling
  !  GALGorithm with the EVOLve keyword from CHARMM.  This evolks a parser
  !  which sets-up the control variables for the genetic algorithm
  !  evolution and then runs the genetic evolution.  The specific functions
  !  performed by this portion of the code are:
  !
  !     1.  Parse parameters controlling the GA evolution.
  !
  !     2.  Evolve the population of chromosomes.  This involves the following
  !         steps:
  !
  !         a) Evaluate the energy of the population at step 0.
  !         b) Rank the population members in accordance with the viability.
  !         c) Choose parents to develop the next generation through crossing.
  !            The parents are chosen as the N_parent most fit from the
  !            population.
  !         d) Cross parents and introduce random muitations into genes,
  !            if elitist model is being used transfer fittest parent into
  !            the new generation.
  !         e) Reconstruct (via IC build-like procedure) the cartesian
  !            positions of the new generation from modified ICs.
  !         f) Begin again at step a), cycle through to convergence or
  !            MAXGenerations.
  !         g) Gather statistics on the population characteristics as
  !            evolution proceeds.
  !

  use chm_kinds
  use dimens_fcm
  use galgor_ltm

  implicit none
  !  data stnructures for genetic 
  !  algorithm searching of conformational space.
  !
#if KEY_GENETIC==1 /*galgor_fcm*/
  !   on* variables are the original counter variables before
  !       replication.  These are used to permit energies of
  !       individual chromosomes to be computed:
  !...old PSF (and IC) counters.
  !========================================================================

  INTEGER onAtom, onRes, onSeg, onBond, onTheta, onPhi,  &
       onImPhi, onNB, onDon, onAcc, onGrp, onST2, olenIC
  !
  !========================================================================
  !...Control data for GA and copies
  !
  !   nChromos            number of active chromosomes
  !   nAtm_chrom          number of atoms per chromosome (should equal
  !                       onAtom)
  !
  INTEGER nChromos, nAtm_chrom
  !
  !   nDihe_IC            number of "active" dihedral ICs
  !   nAnglL_IC           number of "active" angle ICs
  !   nBondL_IC           number of "active" bond ICs
  !   nAnglR_IC           number of "active" angle ICs
  !   nBondR_IC           number of "active" bond ICs
  !   ntran_IC            number of active translational variables
  !   nrota_IC            number of active euler angles
  !   nActive             number of Active variables
  !   Dihe_active         pointer to list of active dihedral IC's
  !   AnglL_active        pointer to list of active angle IC's
  !   BondL_active        pointer to list of active bond IC's
  !   AnglR_active        pointer to list of active angle IC's
  !   BondR_active        pointer to list of active bond IC's
  !   tran_active         pointer to list of active center of mass coordinates
  !   rota_active         pointer to list of active euler angles
  !   idepend             pointer to list of dependent dihedral angles
  !   vdepend             pointer to the values dependent dihedral angles
  !   qdepend             the logical flag determining the dependencies 
  !
  !
  !========================================================================
  INTEGER nDihe_IC, nAnglL_IC, nBondL_IC, nAnglR_IC, nBondR_IC, &
       nAvoid,  nActive , ntran_IC ,nrota_IC, ranktemp(1000)

  integer, allocatable, dimension(:) :: Dihe_active, AnglL_active
  integer, allocatable, dimension(:) :: BondL_active, AnglR_active, BondR_active
  integer, allocatable, dimension(:) :: idepend, tran_active, rota_active
  real(chm_real), allocatable, dimension(:) :: vdepend

  !========================================================================
  !
  !...Control for GA evolution
  !
  !    nGeneration        number of generations to evolve by GA
  !    nParents           number of parents for crossing
  !    GeneSize           number of "bits" to use in gene representation
  !    nNiches            number of niches if niching is used
  !    nCombineNiches     how often niches cross over
  !    Toler              tolerance control variable for GA convergence
  !    MutRate            mutation rate on per gene basis
  !    StepDihe           maximum step size for mutation of dihedral
  !    StepAngl           maximum step size for mutation of angle
  !    StepBond           maximum step size for mutation of bond
  !    StepTran           maximum step size for translational mutation
  !    StepRota           maximum step size for Euler angle
  !========================================================================
  INTEGER nGeneration,  nParents, GeneSize,  &
       nNiches, nCombineNiches, idum,nchildren
  real(chm_real)  Toler, MutRate, &
       StepDihe, StepAngl, StepBond, Width &
       ,StepTran, StepRota
  !     *        ,c0x,c0y,c0z
  !
  ! Coordinates of the point in space acting as an atom for NOE contraints
  !
  !
  !
  !     *        ,c0x,c0y,c0z
  !
  !     rGAmin - minimum distance for the nbonded interactions to be 
  !     computed
  !
  !========================================================================
  !
  !    IC_Rij             signed pointer into IC tables to seed GA builds
  !    IC_Rjk             signed pointer into IC tables to seed GA builds
  !    IC_Tht             signed pointer into IC tables to seed GA builds
  !    is1-is3            pointers to seed atoms for ic builds
  !
  INTEGER IC_Rij, IC_Rjk, IC_Tht, is1, is2, is3
  logical,allocatable,dimension(:) :: qParent

  integer nmonte
  !
  logical Qtran,qrota,qdepend,qavoid,qmonte,QGALGOR,qimproper
  !,qpnoe
  !
  !  qpnoe   Logical flag turning on the constraints to the point in space
  !
  !
  !========================================================================
  !
  ! INTEGER LstAngl, LstPhi, LstImPhi  Moved to galgor_ltm.src by APH 20-May-2011
#endif /* (galgor_fcm)*/
  !

contains
#if KEY_GENETIC==1 /*genetic_main*/

  subroutine galgor_init
    ! called from iniall
    qGA_Ener = .false.
    Qtran = .false.
    Qrota = .false.
    Qmonte = .false.
    Qdepend = .false.
    Qavoid = .false.
    QGALGOR = .false.
    qimproper= .false.
    return
  end subroutine galgor_init

  Subroutine genetic_alg(setup, evolve)
    !
    !  This is the main calling routine for the Genetic Algorithm sampling
    !  procedures.  This routine was written by Charles L. Brooks, III and
    !  Heidi Daigler during the summer of 1994.
    !
    LOGICAL setup, evolve
    !
    !
    !  Begin execution
    IF(setup) then
       CALL GA_setup
    ELSE IF(evolve) then
       qmonte=.false.
       CALL GA_evolve
    Endif
    !
    Return
  End Subroutine genetic_alg
  !

  !
  Subroutine GA_setup
    !  This routine establishes the basic data structures and call the replica
    !  command
    !
  use select
  use intcor_module
  use psf
  use comand
  use stream
  use string
  use bases_fcm
  use coordc
  use memory
    !
    ! Local variables.
    integer,allocatable,dimension(:) :: islct,jslct, Work
    integer,allocatable,dimension(:) :: igrplt, imap, invmap, irglst, isglst
    !      INTEGER LEN

    INTEGER count, howmany, i, ipa(3), nipa
    INTEGER temp_B1IC, temp_B2IC, &
         temp_T1IC, temp_T2IC, temp_PIC
    LOGICAL qVar, qIC, qCoor, qDone
    !
    !  Initialize data structure for GA
    nmonte=0
    onAtom = 0
    onRes = 0
    onSeg = 0
    onBond = 0
    onTheta = 0
    onPhi = 0
    onImPhi = 0
    onNB = 0
    onDon = 0
    onAcc = 0
    onGrp = 0
    onST2 = 0
    olenIC = 0
    nDihe_IC = 0
    nAnglL_IC = 0
    nBondL_IC = 0
    nAnglR_IC = 0
    nBondR_IC = 0
    !      Dihe_active = 0
    !      AnglL_active = 0
    !      BondL_active = 0
    !      AnglR_active = 0
    !      BondR_active = 0
    !      tran_active=0
    !      rota_active=0
    nGeneration = 10
    nParents = 0
    GeneSize = 1
    idum=0
    nNiches = 0
    nCombineNiches = 0
    Toler = 0.0
    MutRate = 0.0
    StepDihe = 0.0
    StepAngl = 0.0
    StepBond = 0.0
    StepTran = 0.0
    StepRota = 0.0
    ntran_IC = 0
    nrota_IC =0
    !
    qVar = .false.
    qCoor = .false.
    qIC = .false.
    qDone = .false.
    count = 0
    qrota=.false.
    qtran=.false.
    !
    ! Parse the command line
    !
    ! the number of chromosomes for population
    nChromos = GtrmI(comLyn,comLen,'CHRO',0)
    nchromos = nchromos*2
    if(prnlev > 1) Write(OutU,'(a,i4,a)') &
         ' GA> Genetic algorithm setup has been called to run ', &
         nChromos,' chromosomes'
    ! the atoms to be replicated for GA sampling
    call chmalloc("genetic.src","ga_setup","islct",natom,intg=islct)
    nAtm_chrom = AtmSel (comLyn, comLen, islct, .FALSE.)
    If ( nAtm_chrom  /=  nAtom ) then
       if(prnlev > 1) Write(OutU, '(a,i4,/,a,i4,a)') &
            ' GA-WARNING> Number of atoms in a chromosome ', nAtm_chrom, &
            ' does not match number of atoms in original system ', &
            nAtom,' proceeding anyway'
    Else
       if(prnlev > 1) Write(OutU, '(a,i4,/,a,i4,a)') &
            ' GA-SETUP> Number of atoms in each chromosome is ', &
            nAtm_chrom
    Endif
    !
    !...Keep track of the original PSF counters. (see psf.f90)
    onAtom  = nAtom
    onRes   = nRes
    onSeg   = nSeg
    onBond  = nBond
    onTheta = nTheta
    onPhi   = nPhi
    onImPhi = nImPhi
    onNB    = nNB
    onDon   = nDon
    onAcc   = nAcc
    onGrp   = nGrp
    onST2   = nST2
    !
    !  Find the seed IC numbers for building Cartesians for chromosomes
    !
    !
    !-----------------------------------------------------------------------
    !  Set default atom numbers to 1-3
    is1=1
    is2=2
    is3=3
    If ( (IndxA(comLyn,comLen,'SEED') > 0) ) then
       !  PROCESS-SEED-COMMAND
       Ipa(1) = 0
       Ipa(2) = 0
       Ipa(3) = 0
       CALL NXTATM(IPA,NIPA,3,COMLYN,COMLEN,ISLCT, &
            SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
       !
       IF( Ipa(1) * Ipa(2)* Ipa(3)  <=  0 ) THEN
          CALL WRNDIE(0,'<GA-SETUP>','ATOM OF SEED DOES NOT EXIST.')
       EndIf
       is1 = Ipa(1)
       is2 = Ipa(2)
       is3 = Ipa(3)
       if(prnlev  >  1) Write(OutU,'(A,I4,A,I4,A,I4,A)') &
            'GA-SETUP> Atoms', is1,',',is2,' and', &
            is3,' used to seed building.'

    Else
       if(prnlev  >  1) Write(OutU,'(A)') &
            'GA-SETUP> No seed specified, will use first three atoms.'
    Endif
    !
    ! now get info about the ICs to sample
    olenIC = icr_struct%lenIC
    !
    !
    qVar  = (IndxA(comLyn,comLen,'VARI') > 0)
    If ( qVar ) then
       call chmalloc("genetic.src","GA_setup","Dihe_active",olenIC,intg=Dihe_active)
       call chmalloc("genetic.src","GA_setup","AnglL_active",olenIC,intg=AnglL_active)
       call chmalloc("genetic.src","GA_setup","BondL_active",olenIC,intg=BondL_active)
       call chmalloc("genetic.src","GA_setup","AnglR_active",olenIC,intg=AnglR_active)
       call chmalloc("genetic.src","GA_setup","BondR_active",olenIC,intg=BondR_active)
       call chmalloc("genetic.src","GA_setup","idepend",olenIC,intg=idepend)
       call chmalloc("genetic.src","GA_setup","idepend",olenIC,crl=vdepend)
       !
       Call GA_getIC(icr_struct%LenIC,ISTRM,ATYPE,IBASE, &
            SEGID,RESID,NICTOT,NSEGT,RES,nAtom,ISLCT, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR, &
            nChromos, nDihe_IC, nAnglL_IC, nBondL_IC, &
            nAnglR_IC, nBondR_IC, &
            ntran_IC,nrota_IC, &
            Dihe_active, AnglL_active, &
            BondL_active, AnglR_active, &
            BondR_active &
            ,qtran,qrota,idepend,vdepend,qdepend &
            ,qimproper)
       !
       !  Now load all IC active selections into GA replicated activity arrays
       !  Get temp array to expedite move
       call chmalloc("genetic.src","ga_setup","work",olenic,intg=work)
       if(prnlev > 1) Write(OutU,'(a)') &
            ' GA-SETUP> Active GA search space will include:'
       If ( nDihe_IC  >  0 ) then
          if(prnlev > 1) Write(OutU,'(a,i4,a/)') &
               '     ', nDihe_IC/nChromos,' dihedral angles'
          nDihe_IC = olenIC * nChromos
          Call MvArray(Work, Dihe_active, olenIC, 1)
          Call MvArray(Dihe_active, Work, olenIC, 1 )

       Else
          call chmdealloc("genetic.src","GA_setup","Dihe_active",olenIC,intg=Dihe_active)
       Endif
       If ( ( nAnglL_IC + nAnglR_IC )  >  0 ) then
          if(prnlev > 1) Write(OutU,'(a,i4,a/)') &
               '     ', (nAnglL_IC+nAnglR_IC)/nChromos,' angles'
          If (nAnglL_IC  >  0) then
             nAnglL_IC = olenIC * nChromos
             Call MvArray(Work, AnglL_active, olenIC, 1)
             Call MvArray( AnglL_active, Work, olenIC, 1 )
          Else
             call chmdealloc("genetic.src","GA_setup","AnglL_active",olenIC,intg=AnglL_active)
          Endif
          If (nAnglR_IC  >  0) then
             nAnglR_IC = olenIC * nChromos
             Call MvArray(Work, AnglR_active, olenIC, 1)
             Call MvArray( AnglR_active, Work, olenIC, 1 )
          Else
             call chmdealloc("genetic.src","GA_setup","AnglR_active",olenIC,intg=AnglR_active)
          Endif
       Endif
       If ( ( nBondL_IC + nBondR_IC )  >  0 ) then
          if(prnlev > 1) Write(OutU,'(a,i4,a/)') &
               '     ', (nBondL_IC+nBondR_IC)/nChromos,' bonds'
          If (nBondL_IC  >  0) then
             nBondL_IC = olenIC * nChromos
             Call MvArray(work, BondL_active, olenIC, 1)
             Call MvArray( BondL_active, work, olenIC, 1 )
          Else
             call chmdealloc("genetic.src","GA_setup","BondL_active",olenIC,intg=BondL_active)
          Endif
          If (nBondR_IC  >  0) then
             nBondR_IC = olenIC * nChromos
             Call MvArray(work, BondR_active, olenIC, 1)
             Call MvArray( BondR_active, work, olenIC, 1 )
          Else
             call chmdealloc("genetic.src","GA_setup","BondR_active",olenIC,intg=BondR_active)
          Endif
       Endif

       If ( ntran_IC  >  0 ) then
          !
          if(prnlev > 1) Write(OutU,'(a,i4,a/)') &
               '     ', ntran_IC/nChromos,' translation'
          ntran_IC = 3 * nChromos
          call chmalloc("genetic.src","GA_setup","tran_active",3,intg=tran_active)
          Call GA_ActiveIC( tran_active, 3, .true., 0 )

       ELSE
          !           tran_active = 0
       ENDIF

       If ( nrota_IC  >  0 ) then
          if(prnlev > 1) Write(OutU,'(a,i4,a/)') &
               '     ', nrota_IC/nChromos,' euler angles'
          nrota_IC = 3 * nChromos
          call chmalloc("genetic.src","GA_setup","rota_active",3,intg=rota_active)
          Call GA_ActiveIC( rota_active, 3, .true., 0 )
       Else
          !           rota_Active = 0
       Endif

    Else

       !  Set default that all dihedrals are active GA variables
       nDihe_IC = nChromos*olenIC
       call chmalloc("genetic.src","GA_setup","Dihe_active",olenIC,intg=Dihe_active)
       Call GA_activeIC(Dihe_active,oLenIC,.true.,0)
       if(prnlev > 1) Write(OutU,'(a)') &
            ' GA-SETUP> Active GA search space will be all dihedrals'
       call chmdealloc("genetic.src","GA_setup","Dihe_active",olenIC,intg=Dihe_active)

    Endif
    !***HLD
    !  Finally, lets replicate the primary system nChromos times

    Call GA_replicate( nChromos, nAtm_chrom, &
         Islct, .true., 'C   ' )

    !  Finally, delete atoms in initial set

    call chmalloc("genetic.src","GA_setup","imap",natom,intg=imap)
    call chmalloc("genetic.src","GA_setup","invmap",natom,intg=invmap)
    call chmalloc("genetic.src","GA_setup","isglst",natom,intg=isglst)
    call chmalloc("genetic.src","GA_setup","irglst",natom,intg=irglst)
    call chmalloc("genetic.src","GA_setup","igrplt",natom,intg=igrplt)

    !
    !  Copy atom selection for deletion to array of correct dimension
    call chmalloc("genetic.src","ga_setup","jslct",natom,intg=jslct)
    Call GA_activeIC( Jslct, nAtom, .false., 0 )
    Call MvArray( Jslct, Islct, onAtom, 1 )
    !
    Call GA_DelAtm(Jslct, iMap, invMap, isGlst, irGlst, iGrpLt)
    call chmdealloc("genetic.src","GA_setup","imap",natom,intg=imap)
    call chmdealloc("genetic.src","GA_setup","invmap",natom,intg=invmap)
    call chmdealloc("genetic.src","GA_setup","isglst",natom,intg=isglst)
    call chmdealloc("genetic.src","GA_setup","irglst",natom,intg=irglst)
    call chmdealloc("genetic.src","GA_setup","igrplt",natom,intg=igrplt)
    !
    ! . Print out the structure file counters.
    CALL PSFSUM(OUTU)
    call chmdealloc("genetic.src","ga_setup","islct",natom,intg=islct)
    call chmdealloc("genetic.src","ga_setup","jslct",natom,intg=jslct)
    call chmdealloc("genetic.src","ga_setup","work",olenic,intg=work)
    Return
  End Subroutine GA_setup


  !***HLD
  Subroutine Hold_IC (B1IC, B2IC, T1IC, T2IC, PIC, target_B1IC, &
       target_B2IC, target_T1IC, target_T2IC, &
       target_PIC, &
       lenIC)

    real(chm_real) B1IC(*), B2IC(*), T1IC(*), T2IC(*), PIC(*), &
         target_B1IC(*), &
         target_B2IC(*), target_T1IC(*), target_T2IC(*), &
         target_PIC(*)

    integer lenIC, i

    DO i = 1, lenIC
       target_B1IC(i) = B1IC(i)
       target_B2IC(i) = B2IC(i)
       target_T1IC(i) = T1IC(i)
       target_T2IC(i) = T2IC(i)
       target_PIC(i)  = PIC(i)
    ENDDO

    Return
  end Subroutine Hold_IC
  !
  INTEGER FUNCTION SUMACT (Active_arr, length)
    ! Sums the values in an array.  Used to determine the number of
    ! active variables.

    INTEGER Active_arr(*), length
    INTEGER i, total

    total = 0
    DO i = 1, length
       total = total + Active_arr(i)
    ENDDO

    SUMACT = total
    Return
  End FUNCTION SUMACT

  !***HLD*END

  Subroutine GA_DelAtm(Islct, Map, InvMap, SegLst, &
       ResLst, GrpLst)
    !  This subroutine takes a list of atoms and deletes them from the PSF.
    !  It is taken from DELTIC
  use memory
  use psf
  use modpsf
  use coord

    INTEGER   ISLCT(:)
    INTEGER   MAP(:), INVMAP(:), SEGLST(:)
    INTEGER   RESLST(:), GRPLST(:)
    !
    !
    integer,allocatable,dimension(:) :: linb, liblo
    INTEGER   I, J, II, OFFSET, K
    !
    !     Temporary mark for deleted atoms:
    INTEGER,PARAMETER :: MARK = -9999

    if (natomt .gt. natom) then
        call wrndie(-5, '<DELTIC>', &
            'GENEtic command not supported with images.')
    end if

    call chmalloc("genetic.src","GA_DelAtm","linb",nnb,intg=linb)
    call chmalloc("genetic.src","GA_DelAtm","liblo",natom,intg=liblo)

    !
    ! construct MAP and INVMAP
    !
    II=NATOM
    DO I=1,NATOM
       IF (ISLCT(I) == 1) THEN
          II=II-1
          MAP(I)=MARK
       ENDIF
    ENDDO
    IF (NATOM == II) THEN
       CALL WRNDIE(0,'<DELTIC>', &
            'No atoms selected for deletion. Nothing done.')
       call chmdealloc("genetic.src","GA_DelAtm","linb",nnb,intg=linb)
       call chmdealloc("genetic.src","GA_DelAtm","liblo",natom,intg=liblo)
       RETURN
    ENDIF
    !
    NATOM=II
    !
    OFFSET=0
    DO I=1,NATOM
       DO WHILE (ISLCT(I+OFFSET) == 1)
          OFFSET=OFFSET+1
       ENDDO
       INVMAP(I)=I+OFFSET
    ENDDO
    DO I=1,NATOM
       MAP(INVMAP(I))=I
    ENDDO
    !
    !     fill temporary segment list
    !
    DO I=1,NSEG
       DO J=NICTOT(I)+1,NICTOT(I+1)
          DO K=IBASE(J)+1,IBASE(J+1)
             SEGLST(K)=I
          ENDDO
       ENDDO
    ENDDO
    !
    !     fill temporary residue list
    !
    DO I=1,NRES
       DO J=IBASE(I)+1,IBASE(I+1)
          RESLST(J)=I
       ENDDO
    ENDDO
    !
    !     fill temporary group list
    !
    DO I=1,NGRP
       DO J=IGPBS(I)+1,IGPBS(I+1)
          GRPLST(J)=I
       ENDDO
    ENDDO
    !
    !     now do mapping and compression of PSF,...
    !
    CALL MAPIC(MAP,INVMAP,SEGLST,RESLST,GRPLST,LINB, &
         LIBLO,MARK,.FALSE.)
    call chmdealloc("genetic.src","GA_DelAtm","linb",nnb,intg=linb)
    call chmdealloc("genetic.src","GA_DelAtm","liblo",natom,intg=liblo)
    !
    CALL CMPRIC(MARK,.false.)
    !
    RETURN
  END Subroutine GA_DelAtm

  Subroutine GA_replicate(nCopy, nSel, Islct, qSetIC, newSeg)
    !  This routine is identical to the soubroutine Replica written by
    !  Leo Caves.  We have stolen it here so we can call the replica stuff
    !  directly for the genetic algorithm set-up

    !-----------------------------------------------------------------------
    !     Generate replicas of arbitrary regions of the psf.
    !     May-9-1991 (Leo Caves)
    !
    !-----------------------------------------------------------------------
  use chm_types
  use psf
  use param
  use code
  use bases_fcm
  use comand
  use stream
  use number
  use replica_mod
  use econtmod
  use memory
    !
    ! Local Pointers.
    integer,PARAMETER :: nPtr=6,pFlg=1,pInd=2,pGrp=3,pRes=4, &
         pFl2=5,pMap=6
    type(chm_iptr),dimension(nptr) :: SRepWk

    ! Local variables.
    CHARACTER(len=*) newSeg
    INTEGER i, nSel, nCopy, onSeg, Islct(*)
    LOGICAL qSetIC
    !
    ! Begin Executable Code.
    !
    !  CHeck and make sure no replicas exist as of yet
    If(qRep) then
       if(prnlev > 1) &
            Write(OutU,'(a)') ' Replicas exist, cannot use GA too!'
       CALL WRNDIE(-1,'<GALG/REPLICA>','REPLICAs exist.')
       RETURN
    Endif

    ! From this point on we need the replica data structures, make sure 
    ! they are allcated and if not allocate them.
    if(.not. allocated(repnoa)) call allocate_replica
    !
    !
    ! the replica code depends on presence of econt arrays, check them
    ! allocate if necessary
    if(.not.allocated(econt)) call allocate_econt
    !
    !


    onSeg = nSeg
    ! Initialize the pointer array for working arrays.
    !------------------------------------------------------
    ! Allocate wk arrays of size nAtom
    DO i = 1, 4
       call chmalloc('genetic.src','GA_replicate','SRepWk(i)',nAtom,intgp=SRepWk(i)%a)
       SRepWk(i)%a(1:nAtom) = 0
    enddo
    ! Allocate wk arrays of size nAtom+1
    DO i = 5, 6
       call chmalloc('genetic.src','GA_replicate','SRepWk(i)',nAtom+1,intgp=SRepWk(i)%a)
       SRepWk(i)%a(1:nAtom+1) = 0
    enddo
    !------------------------------------------------------

    ! the atoms to be replicated.
    !  Copy islct array to pointer array
    Call MvArray(SRepWk(pFlg)%a,islct,nAtom,1)

    !
    !...the new segment identifier
    !...parse this last to be consistent with GENERate command. Allows blank segid.
    !...in the event of a blank segid, the segid will be the replica number
    !      newSeg = NextA4(comLyn,comLen)
    !
    ! Need to check for duplicate segment names here. <=======
    !
    IF (nSel  >  0) THEN
       !
       ! Initialize the replica flags for the primary system
       ! this could be placed in the standard PSF generation routines.
       IF (.NOT. qRep) CALL RepReset

       ! do the stuff...

       ! NB:   nSub is the number of sub-systems
       !       nRepl is the total number of replicas in all sub-systems
       !       nCopy is the number of replicas in the current sub-system
       !
       nSub = nSub + 1

       CALL Repl2( &
            SRepWk(pFlg)%a, SRepWk(pFl2)%a, &
            SRepWk(pInd)%a, SRepWk(pMap)%a, &
            SRepWk(pGrp)%a, SRepWk(pRes)%a, &
            newSeg, nCopy, qSetIC )

       !
       ! need someway of wiping out replica handling
       IF ( .NOT. qRep ) qRep = .TRUE.

       if(prnlev > 1) then
          WRITE(outu, '(A, I0, A, I0, A)') ' GALG/REPLIcate> Segments ', &
               onSeg+1, ' to ', nSeg, ' have been generated.'
          WRITE(outu, '(2A, I0, A, I0)') &
               ' GALG/REPLIcate> Their identifiers are ', newSeg, 1, &
               ' to ', nCopy
       endif
       !
       ! Print out the structure file counters
       ! (this also forces a reset of the lists)
       CALL PsfSum ( outu )

    ELSE
       CALL WrnDie(0,'<GALG/Replica>','No atoms selected')
    ENDIF ! (nSel > 0)

    !...Free up space.
    do i = 1, 6
       if (associated(SRepWk(i)%a)) then
          call chmdealloc('genetic.src','GA_replicate','SRepWk(i)', &
               size(SRepWk(i)%a),intgp=SRepWk(i)%a)
       endif
    enddo

    !...Exit.
    RETURN
  END Subroutine GA_replicate

  Subroutine GA_activeIC(List, Length, qAll,index)
    !  This routine initializes a list of length Length to:
    !
    !         qAll            index           action
    !
    !         true              0      List(i) = 1 (1=1,Length)
    !         true             >0          List(index) = 1
    !         false             0      List(i) = 0 (1=1,Length)
    !         false            >0          List(index) = 0
    !
    INTEGER List(*), Length, index, i
    LOGICAL qAll

    If ( qAll ) then
       If ( index  <=  0 ) then
          Do i = 1, Length

             List(i) = 1

          Enddo
       Else
          List(index) = 1
       Endif
    Else
       If ( index  <=  0 ) then
          Do i = 1, Length

             List(i) = 0

          Enddo
       Else
          List(index) = 0
       Endif
    Endif
    Return
  End Subroutine GA_activeIC

  Subroutine GA_getIC(LenIC,IUNIC,ATYPE,IBASE, &
       SEGID,RESID,NICTOT,NSEG,RES,nAtom,ISLCT, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR, &
       nChromos, nDihe_IC, nAnglL_IC, nBondL_IC, &
       nAnglR_IC, nBondR_IC, &
       ntran_IC,nrota_IC, &
       Dihe_active, AnglL_active, &
       BondL_active, AnglR_active, BondR_active &
       ,qalltr,qallro,idepend,vdepend,Qdepend,qimproper)

    !
    !     EDIT INTERNAL COORDINATES
    !
    !     By Bernard R. Brooks    1982
    !  Modified by C. L. Brooks, III to just choose and label chosen ICs
    !
  use comand
  use stream
  use string
  use select
    !
    INTEGER LENIC,IUNIC
    CHARACTER(len=*) SEGID(*),RESID(*),ATYPE(*),RES(*)
    INTEGER IBASE(*),NICTOT(*)
    INTEGER NSEG,NATOM,ISLCT(*)
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*),qdepend,qimproper,qdep1
    INTEGER nChromos, nDihe_IC, nAnglL_IC, nBondL_IC, &
         nAnglR_IC, nBondR_IC, &
         Dihe_active(*), AnglL_active(*), BondL_active(*), &
         AnglR_active(*), BondR_active(*), &
         ntran_IC,nrota_IC
    real(chm_real) Dihe_radius
    real(chm_real) vdepend(*)
    integer ilast,idepend(*)
    !
    !
    INTEGER NOLD,NNEW,I,J,IIC,IVL,K,L
    INTEGER QAT(4),NQAT
    INTEGER II,INC,ISTART
    !
    CHARACTER(len=4) WRD
    LOGICAL T
    LOGICAL FOUND,KILL,EOF,OK,LPOS,LNEG,DONE
    LOGICAL qEXCLUDE, qALL, qNONE, qAllB, qAllT
    LOGICAL qAllP, qINCLUDE
    LOGICAL qalltr,qallro
    INTEGER COUNT
    !
    !
    EOF=.FALSE.
    NOLD=LENIC
    NNEW=0
    KILL=.FALSE.
    DONE=.FALSE.
    qEXCLUDE = .false.
    qAllB = .false.
    qAllT = .false.
    qAllP = .false.
    qNONE = .false.
    qalltr = .false.
    qallro = .false.
    qdepend = .false.
    qdep1=.false.
    qimproper=.false.
    do iic=1,lenic
       idepend(iic)=0
       vdepend(iic)=0.
    end do
    ilast=-2
    Call GA_ActiveIC( BondL_active, LenIC, .false., 0 )
    Call GA_ActiveIC( BondR_active, LenIC, .false., 0 )
    Call GA_ActiveIC( AnglL_active, LenIC, .false., 0 )
    Call GA_ActiveIC( AnglR_active, LenIC, .false., 0 )
    Call GA_ActiveIC( Dihe_active, LenIC, .false., 0 )
    !
    DO WHILE(.NOT.DONE)
       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIC,EOF,.TRUE., &
            .TRUE.,'GA_getIC> ')
       DO I=1,4
          QAT(I)=0
       ENDDO
       WRD=NEXTA4(COMLYN,COMLEN)
       !-----------------------------------------------------------------------
       IF(WRD == '    ') THEN
          CONTINUE
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'TRAN') THEN
          !           if(.NOT.qAlltr) qAlltr = (IndxA(comLyn,comLen,'ALL') > 0)
          !           qINCLUDE = (IndxA(comLyn,comLen,'INCL') > 0)
          !           qEXCLUDE = (IndxA(comLyn,comLen,'EXCL') > 0)
          qalltr= .true.
          !           If(qAlltr) then
          ntran_IC = 3
          !              Call GA_ActiveIC( tran_active, ntran_IC, .true., 0 )
          !              qAlltr = .false.
          !             Endif
          if(.not.qallro) then
             qallro=(IndxA(comLyn,comLen,'ROTA') > 0)
             if(qallro) nrota_IC = 3
          endif
       ELSE IF(WRD == 'ROTA') THEN
          qallro = .true.
          nrota_IC = 3
          if(.not.qalltr) then
             qalltr=(IndxA(comLyn,comLen,'TRAN') > 0)
             if(qalltr) ntran_IC = 3
          endif
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'BOND') THEN
          if(.NOT.qAllB) qAllB = (IndxA(comLyn,comLen,'ALL') > 0)
          qINCLUDE = (IndxA(comLyn,comLen,'INCL') > 0)
          qEXCLUDE = (IndxA(comLyn,comLen,'EXCL') > 0)
          If(qAllB) then
             nBondL_IC = LenIC
             nBondR_IC = LenIC
             Call GA_ActiveIC( BondL_active, nBondL_IC, .true., 0 )
             Call GA_ActiveIC( BondR_active, nBondR_IC, .true., 0 )
             qAllB = .false.

          Else
             ! process-distance-edit
             CALL NXTATM(QAT,NQAT,2,COMLYN,COMLEN,ISLCT, &
                  SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
             I=QAT(1)
             J=QAT(2)
             !
             IF (I <= 0.OR.J.LE.0) THEN
                !              atom-doesnt-exist
                IF (WRNLEV >= 2) WRITE(OUTU,925)
                CALL DIEWRN(0)
             ELSE
                FOUND=.FALSE.
                DO IIC=1,LENIC
                   IVL=0
                   IF(I == IAR(IIC).OR.J.EQ.IAR(IIC)) IVL=IVL+1
                   IF(I == JAR(IIC).OR.J.EQ.JAR(IIC)) IVL=IVL+2
                   IF(I == KAR(IIC).OR.J.EQ.KAR(IIC)) IVL=IVL+4
                   IF(I == LAR(IIC).OR.J.EQ.LAR(IIC)) IVL=IVL+8
                   IF((IVL == 5.AND.TAR(IIC)).OR. &
                        (IVL == 3.AND..NOT.TAR(IIC))) THEN
                      FOUND=.TRUE.
                      If ( qExclude ) Then
                         nBondL_IC = nBondL_IC - 1
                         Call GA_ActiveIC( BondL_active, &
                              nBondL_IC, .false., IIC )
                      Else
                         nBondL_IC = nBondL_IC + 1
                         Call GA_ActiveIC( BondL_active, &
                              nBondL_IC, .true., IIC )
                      Endif
                      IF(PRNLEV >= 2) WRITE(OUTU,115) IIC
115                   FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
                   ENDIF
                   IF(IVL == 12) THEN
                      FOUND=.TRUE.
                      IF(PRNLEV >= 2) WRITE(OUTU,125) IIC
                      If ( qExclude ) Then
                         nBondR_IC = nBondR_IC - 1
                         Call GA_ActiveIC( BondR_active, &
                              nBondR_IC, .false., IIC )
                      Else
                         nBondR_IC = nBondR_IC + 1
                         Call GA_ActiveIC( BondR_active, &
                              nBondR_IC, .true., IIC )
                      Endif
125                   FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
                   ENDIF
                ENDDO
                !
             Endif
          Endif
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'ANGL') THEN
          ! PROCESS-ANGLE-EDIT
          if(.NOT.qAllT) qAllT = (IndxA(comLyn,comLen,'ALL') > 0)
          qINCLUDE = (IndxA(comLyn,comLen,'INCL') > 0)
          qEXCLUDE = (IndxA(comLyn,comLen,'EXCL') > 0)
          If(qAllT) then
             nAnglL_IC = LenIC
             nAnglR_IC = LenIC
             Call GA_ActiveIC( AnglL_active, nAnglL_IC, .true., 0 )
             Call GA_ActiveIC( AnglR_active, nAnglR_IC, .true., 0 )
             qAllT = .false.
          Else
             CALL NXTATM(QAT,NQAT,3,COMLYN,COMLEN,ISLCT, &
                  SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
             I=QAT(1)
             J=QAT(2)
             K=QAT(3)
             !
             IF(I <= 0.OR.J.LE.0.OR.K.LE.0) THEN
                ! ATOM-DOESNT-EXIST
                IF(WRNLEV >= 2) WRITE(OUTU,925)
                CALL DIEWRN(0)
             ELSE
                FOUND=.FALSE.
                DO IIC=1,LENIC
                   IVL=0
                   IF(I == IAR(IIC).OR.K.EQ.IAR(IIC)) IVL=IVL+1
                   IF(I == JAR(IIC).OR.K.EQ.JAR(IIC)) IVL=IVL+2
                   IF(I == KAR(IIC).OR.K.EQ.KAR(IIC)) IVL=IVL+4
                   IF(I == LAR(IIC).OR.K.EQ.LAR(IIC)) IVL=IVL+8
                   IF(.NOT.((IVL /= 3.OR.J.NE.KAR(IIC).OR..NOT.TAR(IIC)) &
                        .AND.(IVL /= 5.OR.J.NE.JAR(IIC).OR.TAR(IIC)))) THEN
                      FOUND=.TRUE.
                      If ( qExclude ) Then
                         nAnglL_IC = nAnglL_IC - 1
                         Call GA_ActiveIC( AnglL_active, &
                              nAnglL_IC, .false., IIC )
                      Else
                         nAnglL_IC = nAnglL_IC + 1
                         Call GA_ActiveIC( AnglL_active, &
                              nAnglL_IC, .true., IIC )
                      Endif
                      IF(PRNLEV >= 2) WRITE(OUTU,215) IIC
215                   FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
                   ENDIF
                   IF(IVL == 10.AND.J.EQ.KAR(IIC)) THEN
                      FOUND=.TRUE.
                      If ( qExclude ) Then
                         nAnglR_IC = nAnglR_IC - 1
                         Call GA_ActiveIC( AnglR_active, &
                              nAnglR_IC, .false., IIC )
                      Else
                         nAnglR_IC = nAnglR_IC + 1
                         Call GA_ActiveIC( AnglR_active, &
                              nAnglR_IC, .true., IIC )
                      Endif
                      IF(PRNLEV >= 2) WRITE(OUTU,225) IIC
225                   FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
                   ENDIF
                ENDDO
                !
             ENDIF
          Endif
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'DIHE') THEN
          ! PROCESS-DIHEDRAL-EDIT
          !
          !
          if(.NOT.qAllP) qAllP = (IndxA(comLyn,comLen,'ALL') > 0)
          qNONE = (IndxA(comLyn,comLen,'NONE') > 0)
          qINCLUDE = (IndxA(comLyn,comLen,'INCL') > 0)
          qEXCLUDE = (IndxA(comLyn,comLen,'EXCL') > 0)
          qdep1 = (IndxA(comLyn,comLen,'DEPE') > 0)
          if(qdep1) qdepend=.true.
          qimproper = (IndxA(comLyn,comLen,'IMPR') > 0)
          If(qAllP) then
             nDihe_IC = LenIC
             Call GA_ActiveIC( Dihe_active, nDihe_IC, .true., 0 )
             qAllP = .false.
          Else if(qNONE) then
             nDihe_IC = 0
          Else
             CALL NXTATM(QAT,NQAT,4,COMLYN,COMLEN,ISLCT, &
                  SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
             I=QAT(1)
             J=QAT(2)
             K=QAT(3)
             T=(K < 0)
             IF(T) K=-K
             L=QAT(4)
             !
             IF(I <= 0.OR.J.LE.0.OR.K.LE.0.OR.L.LE.0) THEN
                ! ATOM-DOESNT-EXIST
                IF(WRNLEV >= 2) WRITE(OUTU,925)
                CALL DIEWRN(0)
             ELSE
                !
                FOUND=.FALSE.
                DO IIC=1,LENIC
                   IF(I == IAR(IIC).AND.L.EQ.LAR(IIC)) THEN
                      LPOS=(J == JAR(IIC).AND.K.EQ.KAR(IIC))
                      LNEG=(J == KAR(IIC).AND.K.EQ.JAR(IIC))
                   ELSE
                      IF(I == LAR(IIC).AND.L.EQ.IAR(IIC)) THEN
                         LNEG=(J == JAR(IIC).AND.K.EQ.KAR(IIC))
                         LPOS=(J == KAR(IIC).AND.K.EQ.JAR(IIC))
                      ELSE
                         LNEG=.FALSE.
                         LPOS=.FALSE.
                      ENDIF
                   ENDIF
                   !
                   IF(LNEG) THEN
                      IF(PRNLEV >= 2) WRITE(OUTU,315) IIC
315                   FORMAT(15X,'FOUND DIHEDRAL IN IC', &
                           I5,' AS OPPOSITE')
                      IF(T.NEQV.TAR(IIC).AND.WRNLEV >= 2) &
                           WRITE(OUTU,316)
316                   FORMAT(20X,'BUT TYPE VALUES DONT MATCH')
                      IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
                      If ( qExclude ) Then
                         nDihe_IC = nDihe_IC - 1
                         Call GA_ActiveIC( Dihe_active, &
                              nDihe_IC, .false., IIC )
                         idepend(iic)=0
                      Elseif (qInclude) then
                         nDihe_IC = nDihe_IC + 1
                         Call GA_ActiveIC( Dihe_active, &
                              nDihe_IC, .true., IIC )
                         idepend(iic)=-1
                         ilast=iic
                      ELSEif(qimproper) then
                         nDihe_IC = nDihe_IC + 1
                         Call GA_ActiveIC( Dihe_active, &
                              nDihe_IC, .true., IIC )
                         idepend(iic)=-2
                         ilast=iic
                      elseif(qdep1) then
                         !
                         ! dependencies
                         idepend(iic)=ilast
                         vdepend(iic)=GTrmF(Comlyn,Comlen,'OFFS',vdepend(iic))

                      Endif
                   ENDIF
                   IF(LPOS) THEN
                      IF(PRNLEV >= 2) WRITE(OUTU,325) IIC
                      IF(T.NEQV.TAR(IIC) .AND. WRNLEV >= 2) &
                           WRITE(OUTU,316)
                      If ( qExclude ) Then
                         nDihe_IC = nDihe_IC - 1
                         Call GA_ActiveIC( Dihe_active, &
                              nDihe_IC, .false., IIC )
                         idepend(iic)=0
                      Else if(qinclude) then
                         nDihe_IC = nDihe_IC + 1
                         Call GA_ActiveIC( Dihe_active, &
                              nDihe_IC, .true., IIC )
                         idepend(iic)=-1
                         ilast=iic
                      ELSEif(qimproper) then
                         !                       Else if(qinclude) then
                         nDihe_IC = nDihe_IC + 1
                         Call GA_ActiveIC( Dihe_active, &
                              nDihe_IC, .true., IIC )
                         ilast=iic
                         !
                         ! dependencies
                         idepend(iic)=-2
                      ELSEif(qdep1) then
                         idepend(iic)=ilast
                         vdepend(iic)=GTrmF(Comlyn,Comlen,'OFFS',vdepend(iic))
                      Endif
                      IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
325                   FORMAT(15X,'FOUND IN IC',I5,' AS POSITIVE')
                   ENDIF
                ENDDO

             ENDIF
             !
             ! now the dependencies
             !
          Endif
          do iic=1, lenic
             !         write(outu,'(a,i5,2x,i5,2x,f12.4)')'DIHE dependencies ',iic,
             !     f    idepend(iic)
             !     f    ,vdepend(iic)
          end do
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'END ') THEN
          !  Get full length of modified GA/IC list
          ntran_IC = nChromos * ntran_IC
          nrota_IC = nChromos * nrota_IC
          nBondL_IC = nChromos * nBondL_IC
          nBondR_IC = nChromos * nBondR_IC
          nAnglL_IC = nChromos * nAnglL_IC
          nAnglR_IC = nChromos * nAnglR_IC
          nDihe_IC = nChromos * nDihe_IC
          DONE = .TRUE.
          !
          !-----------------------------------------------------------------------
       ELSE
          IF(WRNLEV >= 2) WRITE(OUTU,39) WRD
39        FORMAT(' ** UNRECOGNIZED OPERATION "',A4,'" IN EDITIC  **'/)
          CALL DIEWRN(0)
          !-----------------------------------------------------------------------
       ENDIF
       !
    ENDDO
    RETURN
    !yw
375 FORMAT(15X,'DIHEDRAL NOT FOUND. ADDING NEW IC',I5)
925 FORMAT(/15X,'**** ERROR. ATOM OF INTERNAL COORDINATE DOES NOT', &
         ' EXIST. IGNORING ****'/)
    !
  END Subroutine GA_getIC

  Subroutine MvArray1(target, nstart, nend,natom,natomt,num)
    !  This subroutine moves one integer array into another
    INTEGER target(*), natom, nstart, nend,i,num,natomt
    if(num == 1) then
       do i=1, natom
          target(i)=1
       end do
       do i=natom+1,natomT
          target(i)=0
       end do
    endif
    Do i=nstart, nend
       target(i) = 0
    Enddo
    Return
  End Subroutine MvArray1

  Subroutine MvArray(target, source, length, ncopy)
    !  This subroutine moves one integer array into another
    INTEGER target(*), source(*), length, ncopy
    INTEGER i, icount, offset

    If ( ncopy  >  0 ) then
       Do icount = 1, ncopy
          offset = (icount-1)*length
          Do i=1, length
             target(i+offset) = source(i)
          Enddo
       Enddo
    Else
       offset = -ncopy
       Do i=1, length
          target(i+offset) = source(i)
       Enddo
    Endif
    Return
  End Subroutine MvArray

  Subroutine Array_init(target, length)
    !  This subroutine moves one integer array into another
    INTEGER target(*),  length
    INTEGER i
    Do i=1, length
       target(i) = 0
    Enddo

    Return
  End Subroutine Array_init
  !***HLD

  Subroutine MvActive(target, source, active_index, start, length)
    !  This subroutine moves the active elements from the source array into the
    !  target array.  The firstactive element will be placed in <start>+1
    !
    real(chm_real) target(*), source(*)
    INTEGER active_index(*), start, length
    ! local
    INTEGER i, count

    count = 0
    Do i = 1, length
       If (active_index(i) > 0) THEN
          count = count + 1
          target(start+count) = source(i)
       Endif
    Enddo

    Return
  End Subroutine MvActive
  !***HLD*END


  Subroutine GA_evolve
    !  This routine parses the command lines associated with the control of the
    !  GA evolution and disperses these commands to the subroutine GA_cntrl, which
    !  provides overall control for the GA searching.
    !
    !  This routine was written by C. L. Brooks and H. Daigler during the summer
    !  of 1994.
  use intcor_module
  use comand
  use stream
  use string
  use psf
  use bases_fcm
  use coord
  use energym
  use deriv
    !
  use reawri
  use cnst_fcm
  use block_ltm
  use number
  use consta
  use inbnd
  use rndnum
  use memory
  use parallel
  use clcg_mod,only:rngmodseeds

    ! This is an automatically allocated array.
    real(chm_real) GA_Energy(nChromos) ! The energy of chromosomes
    ! local
    !
    INTEGER iGeneration, iCopy, iPrint,i
    logical,allocatable,dimension(:) :: already_ranked
    real(chm_real) ELast, ECurrent
    real(chm_real) tstart,tend,kt1,tincr,temp,pint1,epres,dpres,ktb
    real(chm_real) xt1,yt1,zt1
    integer np1, nprint1
    LOGICAL Converged, qTemp
    LOGICAL qInitialize, qPrntEnergy,qconv
    LOGICAL qconve,qdelete,qclear,qsteady
    LOGICAL QDihe, QImDihe, QBond, QUreyB, QAngle, qpresent
    !
    INTEGER nGenes,nupdate
    INTEGER ncycle,ndelete
    integer,allocatable,dimension(:) :: active_pointer, where_active
    integer,allocatable,dimension(:) :: chromos_rank, acce
    real(chm_real),allocatable,dimension(:) :: ntran, prob, nterm, nrota
    real(chm_real),allocatable,dimension(:,:) :: chromos

    INTEGER iActive, icount, integer1

    INTEGER  iBegin, iSkip, Unit
    real(chm_real) well_dev,crosrate,rrota,rtran,rdihe
    real(chm_real) tolco,offtra,rsame,step_improp,pall
    real(chm_real) bsteptr,bstepro,bstepdih,tbigstep
    real(chm_real) steptr1,stepro1,stepdih1,pcall,temp_store,ptall
    integer ibigstep,ijwalk
    INTEGER start_dihed, start_anglR, start_anglL, &
         start_bondL, start_bondR &
         ,start_TRan, start_Rota
    integer elite,nevaluat,ievaluat,ione
    logical qgenerational,quser,qchar
    integer,allocatable,dimension(:) :: lrngseeds
    !
    ! initialization of variables
    !
    qgalgor=.true.
    do i=1, natom
       dx(i)=0.
       dy(i)=0.
       dz(i)=0.
    end do
#if KEY_BLOCK==1
    NOFORC=.true.  
#endif
    !
    !  Set the defaults for GA evolution control variables
    !
    ! These variables are parsed later
    !
    nGeneration = 0
    nParents = nChromos/2
    GeneSize = 1
    nevaluat=1500000
    ievaluat=0
    nNiches = 1
    nCombineNiches = 0
    Toler = 0.0
    TOLCO= 0.0
    StepDihe = 30.0
    StepAngl = 2.0
    StepBond = 0.002
    xt1=0.
    yt1=0.
    zt1=0.
    pall=0.0
    pcall=0.0
    ptall=0.
    ibigstep=10000000
    ijwalk=10000000
    !      rgamin=0.0
    tbigstep=400
    Rrota=0.0
    Rtran=0.0
    Rdihe=0.0
    offtra=0.2
    StepTran = 0.6
    Step_Improp = 2.0
    StepRota = 0.5
    !      qgamin=.false.
    qconve = .false.
    qconv = .false.
    qdelete = .false.
    ndelete=0
    qclear= .false.
    qsteady = .false.
    qgenerational = .false.
    tstart=300.
    tend=300.
    tincr=0
    pint1=0.5
    elite = 1
    epres=1.1
    nupdate=10
    nprint1=5
    iPrint = 100
    qPrntEnergy = .false.
    qInitialize = .false.
    Unit = 6
    mutrate=1.0
    crosrate=0.0
    nNiches=1
    nchildren=2
    if(ntran_ic > 0 ) qtran=.true.
    if(nrota_ic > 0) qrota=.true.
    !
    !
    nmonte=nmonte+1
    start_dihed = 1
    start_anglL = nDihe_IC/nChromos + 1
    start_anglR = start_anglL + nAnglL_IC/nChromos
    start_bondL = start_anglR + nAnglR_IC/nChromos
    start_bondR = start_bondL + nBondL_IC/nChromos
    start_TRAN = start_bondR + nBondR_IC/nChromos
    start_ROTA = start_TRAN + nTRAN_IC/nChromos
    !
    !  Now parse the command line
    !
    if(prnlev > 1) &
         Write(OutU, '(/a/)') ' GA_Evolve>  Beginning GA evolution'
    qsteady = (IndxA(comLyn,comLen,'STEA') > 0)
    qgenerational = (IndxA(comLyn,comLen,'GENU') > 0)
    Qmonte= (IndxA(comLyn,comLen,'MCAR') > 0)
    !

    !
    !
    !
    nGeneration = GTrmI(Comlyn,Comlen,'MAXG', nGeneration)
    ibigstep= nGeneration +10
    ijwalk= nGeneration +10
    ibigstep=GTrmI(Comlyn,Comlen,'IBIG', ibigstep)
    nprint1=GTrmI(Comlyn,Comlen,'NPRI', nprint1)
    !      rgamin=GTrmf(Comlyn,Comlen,'RMIN', rgamin)
    !      if(rgamin > 0) qgamin=.true.
    ijwalk=GTrmI(Comlyn,Comlen,'IJWA', ijwalk)
    nevaluat=GTrmI(Comlyn,Comlen,'NEVA', nevaluat)
    tbigstep=GTrmf(Comlyn,Comlen,'TJWA', tbigstep)
    ncycle=nGeneration
    nParents = GTrmI(Comlyn,Comlen,'PARE', nParents)
    nchildren=nchromos/2
    nchildren = GTrmI(Comlyn,Comlen,'CHIL', nchildren)
    !
    ! there can never be more children than there are parents
    !
    nchildren=min(nchildren,nchromos-nparents)

    elite = GTrmI(Comlyn,Comlen,'ELIT', elite)
    nupdate = GTrmI(Comlyn,Comlen,'NBFR', nupdate)
    epres = GTrmF(Comlyn,Comlen,'EPRE', epres)
    pall= GTrmF(Comlyn,Comlen,'PALL', pall)
    pcall = GTrmF(Comlyn,Comlen,'PCAL', pcall)
    ptall = GTrmF(Comlyn,Comlen,'PTAL', ptall)
    pall=pall+pcall
    ptall=ptall+pall
    if(epres == 2) epres=2.01
    np1=nparents/nniches
    dpres=(1-epres)*50.*np1
    dpres=dpres/((np1-1)*0.5*epres-np1+1)

    !
    GeneSize = GTrmI(Comlyn,Comlen,'GSIZ', GeneSize)
    nNiches = GTrmI(Comlyn,Comlen,'NICH', nNiches)
    nCombineNiches = GTrmI(Comlyn,Comlen,'INTE', nGeneration+1)
    !

    if(qmonte) then
       elite = 0
       qgenerational=.false.
       qsteady=.false.
       nniches=1
    endif
    !
    if(nNiches == 1) nCombineNiches=nGeneration+1
    nchildren =min(nparents-elite,nchildren)
    if(qgenerational) nchildren=nparents-elite
    if(nniches > 1) then
       i=nparents/nniches
       nparents=i*nniches
       elite=elite*nniches
       if(qgenerational) then
          nchildren=nparents-elite
       else
          i=nchildren/nniches
          nchildren=i*nniches
          if(qsteady) nchildren =min(nparents-elite,nchildren)
       endif
    endif
    !
    !
    !
    if(prnlev > 1) then
       if(qmonte) then
          write(outu,'(a/)') &
               ' MONTE CARLO : Sampling from Boltzmann Distribution '
       else
          if(qsteady.and.nchildren > 2) write(outu,'(a/)') &
               ' GA_Evolve: Generational update, selection pressure from parents'
          if(qsteady.and.nchildren == 2) write(outu,'(a/)') &
               ' GA_Evolve: Steady state update, selection pressure from parents'
          if(.not.(qsteady.or.qgenerational)) write(outu,'(a/a/)') &
               ' GA_Evolve: EVOLUTIONARY STRATEGY: select the best members after mating;', &
               '            selection pressure from children and parents'
       endif
    endif
    !
    !

    !
    if(.not.qmonte) then
       if(prnlev > 1)then
          Write(OutU, '(a,i5,a,i5,a,/,a,i5,a)') &
               ' GA_Evolve: There will be ', nNiches, &
               ' niches that contain ', nparents/nniches, ' parents', &
               '            and ', nchildren/nniches, &
               ' children will be created in each'
          Write(OutU, '(a,i5,a,/,a,i5,a,i5,a,/,a,i5,a)') &
               ' GA_Evolve: GA evolution to be carried out for ', &
               nGeneration ,' generations', &
               ' GA_Evolve: The population will consist of ', nChromos , &
               ' of which ', &
               nParents ,' will be parents', &
               '            and',nchildren,' will be children '
          !
          write(outu,'(a,i6)') &
               ' GA_Evolve: The elitism of the population is ', &
               elite/nniches
          write(outu,'(a,f12.4)') &
               ' GA_Evolve: The Evolutionary pressure of is ', &
               epres
          !
          Write(OutU, '(a,i5,a,/,a,i5,a,i5,a)') &
               ' GA_Evolve: Each gene will be represented by ', GeneSize , &
               ' bits of information in the GA.', &
               ' GA_Evolve: There will be ', nNiches, &
               ' niche(s) that combine every ', &
               nCombineNiches, ' generations.'
       endif
    else
       if (prnlev > 1) Write(OutU, '(a,i5,a,/,a,i5)') &
            ' GA_Evolve: MC scheme to be carried out for ', &
            nGeneration ,' generations', &
            ' GA_Evolve: The population will consist of ', nChromos
    endif

    !
    ! end of printing section
    !
    CrosRate = GTrmF(Comlyn,Comlen,'CROS', Crosrate)
    MutRate = GTrmF(Comlyn,Comlen,'MUTA', MutRate)
    StepDihe = GTrmF(Comlyn,Comlen,'DIST', StepDihe)
    Step_improp = GTrmF(Comlyn,Comlen,'IMPR', Step_improp)
    StepAngl = GTrmF(Comlyn,Comlen,'ANST', StepAngl)
    StepBond = GTrmF(Comlyn,Comlen,'BOST', StepBond)
    !
    StepTran = GTrmF(Comlyn,Comlen,'TRAN', StepTran)
    StepRota = GTrmF(Comlyn,Comlen,'ROTA', StepRota)
    bStepDih = GTrmF(Comlyn,Comlen,'BDIS', StepDihe)
    bstepTr = GTrmF(Comlyn,Comlen,'BTRA', StepTran)

    bstepRo = GTrmF(Comlyn,Comlen,'BROT', bStepRo)
    !

    PINT1= GTrmF(Comlyn,Comlen,'PINT', pint1)
    NCYCLE= GTrmi(Comlyn,Comlen,'TFRQ', NCYCLE)
    ncycle=min(ncycle,ngeneration)
    Toler = GTrmF(Comlyn,Comlen,'TOLE', Toler)
    Tolco = GTrmF(Comlyn,Comlen,'TOLC', Tolco)
    if(toler > 0) qconve=.true.
    if(tolco > 0) qconv=.true.
    !      qconv= (IndxA(comLyn,comLen,'CONV') > 0)
    !      qconve= (IndxA(comLyn,comLen,'CONE') > 0)
    qdelete = (IndxA(comLyn,comLen,'DELE') > 0)
    qclear = (IndxA(comLyn,comLen,'CLEA') > 0)
    !
    if(qmonte) then
       !
       !  for Monte Carlo some parameters are different
       !
       nchildren=nchromos/2
       nparents=nchromos/2
       nprint1=nparents
       !
       !  printing the information
       !
    else
       if(prnlev > 1) then
          Write(OutU, '(a,f10.4)') &
               ' GA_Evolve: Mutations will be performed at a rate of ', &
               MutRate

       endif
       !
       mutrate=mutrate+crosrate
       rsame=1-mutrate
       rsame=rsame+mutrate
       if(prnlev > 1) &
            Write(OutU, '(a,f10.4)') &
            ' GA_Evolve: Crossover will be performed at a rate of ', &
            CrosRate
       if (prnlev > 1) write(outu,'(a,i10,a)') &
            ' GA_Evolve: Large scale move will be tried every ', &
            ibigstep,' generations'
    endif
    Tstart = GTrmF(Comlyn,Comlen,'TBEG', Tstart)
    Tend = GTrmF(Comlyn,Comlen,'TEND', Tend)


    qInitialize = (IndxA(comLyn,comLen,'RAND') > 0)
    if(qInitialize) then
       Rdihe = GTrmF(Comlyn,Comlen,'RDIH', Rdihe)
       RTran = GTrmF(Comlyn,Comlen,'RTRA', Rtran)
       xt1=GTrmF(Comlyn,Comlen,'RXTR', xt1)
       yt1=GTrmF(Comlyn,Comlen,'RYTR', yt1)
       zt1=GTrmF(Comlyn,Comlen,'RZTR', zt1)
       offtra = GTrmF(Comlyn,Comlen,'OFTR', offtra)
       Rrota = GTrmF(Comlyn,Comlen,'RROT', Rrota)
       !      iseed = GTrmI(Comlyn,Comlen,'ISEE', iseed)
       !         if (.not.qoldrng) then      !yw 05-Aug-2008
       !            CALL CLCGINIT(ISEED)
       !            ISEED=1
       !         endif
       !
       call chmalloc('genetic.src','GA_Evolve','lrngseeds',Nrand,intg=lrngseeds)
       lrngseeds(1:nrand)=rngseeds(1:nrand)
       if(qoldrandom.or.qbrokenclcg)lrngseeds(1:nrand)=iseed
       call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
       call rngmodseeds(qpresent,iseed)
       call chmdealloc('genetic.src','GA_Evolve','lrngseeds',Nrand,intg=lrngseeds)
       !

    endif
    iPrint = GTrmI(Comlyn,Comlen,'PRIN', iPrint)
    ndelete=nparents
    ndelete = GTrmI(Comlyn,Comlen,'LEAV', ndelete )
    if(qmonte) ndelete=nparents
    !
    ! Printing begins
    StepDihe = StepDihe/GeneSize
    StepAngl = StepAngl/GeneSize
    StepBond = StepBond/GeneSize
    StepTran = StepTran/GeneSize
    StepRota = StepRota/GeneSize
    !
    if(prnlev > 1) then
       write(outu,'(a,a,f12.4)') &
            ' GA_Evolve: Mutation probability of internal degrees ', &
            'of freedom ', pint1
       write(outu,'(a,a,f12.4)') &
            ' GA_Evolve: Mutation probability of all internal genes ', &
            'simultaneously ', (ptall-pall)*pint1
       if(ntran_IC+nrota_ic > 0) then
          write(outu,'(a,f12.4)') &
               ' GA_Evolve: Mutation probability of rigid body genes of is ', &
               (1.-pint1)
          write(outu,'(a,a,f12.4)') &
               ' GA_Evolve: Mutation probability of entire tran or ', &
               'rota gene is ', (pall-pcall)*(1.-pint1)
          write(outu,'(a,a,f12.4)') &
               ' GA_Evolve: Mutation probability of all tran and rota ', &
               'genes is ', pcall*(1.-pint1)
       endif
       Write(OutU, '(a,3f10.4)') &
            ' GA_Evolve: Mutation sizes for bonds, angles and dihedrals: ', &
            StepBond/2., StepAngl/2., StepDihe/2.
       if(ntran_ic+nrota_ic > 0) Write(OutU, '(a,a,3f10.4)') &
            ' GA_Evolve: Mutation sizes for rigid body translation, ', &
            'rotations', StepTran/2., StepRota*90/twopi
    endif
    !
    If ( qInitialize ) then
       if (prnlev > 1) Write(OutU, '(a)') &
            ' GA_Evolve: ICs will be randomized about initial values'

       if(rtran > 0 .and. prnlev > 1) Write(OutU, '(a/,a,3f8.2/,a,f8.2/,a,f8.2)') &
            '            Centers of masses will be distributed in a shell' &
            ,'            centered around',xt1,yt1,zt1, &
            '            with radius of', rtran, &
            '            and exclusion from center of ',offtra
       if(nrota_ic > 0.and.rrota.gt.0 .and. prnlev > 1) &
            Write(OutU, '(a,a,f8.2,a,f8.2)') &
            '            Euler angles will be distributed randomly ', &
            'between',-rrota*90./twopi, ' and',rrota*90./twopi
       if(rrota > 0 .and. prnlev > 1) Write(OutU, '(a,a,f8.2,a,f8.2)') &
            '            Dihedral angles will be distributed randomly ', &
            'between',-rdihe/2., ' and',rdihe/2.
       !
    endif
    if(qgamin) then
       if(prnlev > 1) Write(OutU, '(a,f12.4)') &
            ' GA_Evolve: NBONDS will be truncated at sigma - ', &
            rgamin
       !       rgamin=rgamin**2
    endif
    if (prnlev > 1) Write(OutU, '(a,i5,a)') &
         ' GA_Evolve: Chromosome energies will be printed every', &
         iPrint,' steps.'
    if(qconv .and. prnlev > 1) &
         Write(OutU, '(a,a,f10.4)') &
         ' GA_Evolve: Convergence for coordinate population ', &
         'deviation less than ', &
         Tolco
    if(qconve .and. prnlev > 1) then
       Write(OutU, '(a,a,f10.4)') &
            ' GA_Evolve: Convergence for energy population deviation ', &
            'less than ', &
            Toler
    else
       if(.not.qconv .and. prnlev > 1) write(OutU, '(a)') &
            ' GA_Evolve: Convergence will not be checked.'
    endif
    if(qdelete.and.(.not.qmonte) .and. prnlev > 1) &
         Write(OutU, '(a,i5,a,i5,a,i5/)') &
         ' GA_Evolve: The number of chromosomes to be left  ', &
         ndelete,' per niche:  ', ndelete/nniches,' and deleted', &
         nchromos-ndelete

    if(qmonte) then
       tincr=(tend-tstart)/(float(ngeneration)/ncycle)

       if(prnlev > 1) then
          Write(OutU, '(a,3f10.4)') &
               ' GA_Evolve: Monte Carlo: Starting temerature            : ', &
               Tstart
          Write(OutU, '(a,3f10.4)') &
               ' GA_Evolve: Monte Carlo: Final temerature               : ', &
               Tend
          Write(OutU, '(a,f10.4,2x,i5/)') &
               ' GA_Evolve: Monte Carlo: Temperature increment;frequency : ', &
               tincr, ncycle
       endif
    endif
    !
    ! Printing ends
    !
    !
    !     Begining of setup
    !    SETUP SETUP
    !
    Call GA_Seed(is1,is2, is3, oLenIC, &
         icr_struct%B1ic,icr_struct%B2ic, &
         icr_struct%T1ic,icr_struct%T2ic, &
         icr_struct%PIC, icr_struct%IAR, &
         icr_struct%JAR, icr_struct%KAR, &
         icr_struct%LAR, icr_struct%TAR, &
         IC_Rij, IC_Rjk, IC_Tht)
    !      endif
    !
    !  Set up logical energy flags to control what energy terms are computed
    !  don't compute energy changes for variables which don't change in GA
    !  evolution
    If ( nDihe_IC  <=  0 ) then
       QDihe = QETerm(Dihe)
       QETerm(Dihe) = .false.
    EndIf
    If ( nDihe_IC  <=  0 ) then
       QImDihe = QETerm(ImDihe)
       QETerm(ImDihe) = .false.
    EndIf
    If ( ( nAnglL_IC + nAnglR_IC )  <=  0 ) then
       QAngle = QETerm(Angle)
       QETerm(Angle) = .false.
    EndIf
    If ( ( nAnglL_IC + nAnglR_IC )  <=  0 ) then
       QUreyB = QETerm(UreyB)
       QETerm(UreyB) = .false.
    EndIf
    If ( ( nBondL_IC + nBondR_IC )  <=  0 ) then
       QBond = QETerm(Bond)
       QETerm(Bond) = .false.
    EndIf
    !
    ! Set the energy flags for user energy and harmonic energy
    !
    qchar=Qeterm(charm)
    qeterm(charm)=.false.
    do i=1, onatom*nchromos
       if(kcnstr(i) /= 0) qeterm(charm)=.true.
    end do
    quser=qeterm(user)
    qeterm(user)=.false.
    !

    !*******************Begin evolution**************************************

    !  Allocate space for logical qParent array
    call chmalloc("genetic.src","GA_evolve","qParent",nChromos,log=qParent)
    iGeneration = 1
    qGA_Ener = .true.
    Converged = .false.
    ELast = 0.0

    !  Allocate space for arrays needed in GA_Move, two main arrays, one to
    !  hold the genetic information for the chromosomes and the other to
    !  provide the ranking order for the population

    nGenes =  ( (nDihe_IC + nAnglL_IC + nAnglR_IC + nBondL_IC &
         + nBondR_IC + ntran_IC + nrota_IC)*Genesize ) / nChromos

    integer1=int(nGenes/GeneSize)
    !
    !
    ! BEGINING OF MEMORY ALOCATION (only the first time EVOLVE is accessed)
    !
    !

    call chmalloc("genetic.src","GA_evolve","active_pointer",integer1,intg=active_pointer)

    active_pointer(:) = 0

    ione=1
    If ( nDihe_IC  >  0 )  then
       Call MvArray(active_pointer, Dihe_Active, nDihe_IC/nChromos, ione)
    Endif
    If ( nAnglL_IC  >  0 )  then
       Call MvArray(active_pointer, AnglL_Active, nAnglL_IC/nChromos, -(nDihe_IC/nChromos))
    Endif
    If ( nAnglR_IC  >  0 )  then
       Call MvArray(active_pointer, AnglR_Active, nAnglR_IC/nChromos, &
            -((nDihe_IC+nAnglL_IC)/nChromos))
    Endif
    If ( nBondL_IC  >  0 )  then
       Call MvArray(active_pointer, BondL_Active, nBondL_IC/nChromos, &
            -((nDihe_IC+nAnglL_IC+nAnglR_IC)/nChromos))
    Endif
    If ( nBondR_IC  >  0 )  then
       Call MvArray(active_pointer, BondR_Active, &
            nBondR_IC/nChromos, -(nDihe_IC+nAnglL_IC+nAnglR_IC &
            +nBondL_IC)/nChromos)
    Endif
    !
    If ( ntran_IC  >  0 )  then
       Call MvArray(active_pointer, Tran_Active, &
            ntran_IC/nChromos,  -(nDihe_IC+nAnglL_IC+nAnglR_IC &
            +nBondL_IC+nbondR_IC)/nChromos)
    Endif
    If ( nrota_IC  >  0 )  then
       Call MvArray(active_pointer, Rota_Active, &
            nrota_IC/nChromos, -(nDihe_IC+nAnglL_IC+nAnglR_IC &
            +nBondL_IC+nbondR_IC+ntran_IC)/nChromos)
    Endif
    !
    !
    ! End of memory alocation - stage 1
    !
    !
    !  Determine the number of active variables
    !
    nActive = 0
    Do iActive = 0, nGenes/GeneSize-1
       If ( active_pointer(iActive+1) == 1 ) nActive = nActive + 1
    Enddo
    if(prnlev > 1) then
       Write(OutU,'(a,i5,a)') &
            ' GA_Evolve: Number of active variables is: ', nActive, &
            ' specifically:'
       if(ntran_ic+nrota_ic > 0) &
            Write(OutU,'(a,i5)') &
            ' GA_Evolve: Number of rigid body degrees of freedom: ', &
            ntran_ic/nchromos+nrota_ic/nchromos

       Write(OutU,'(a,i5//)') &
            ' GA_Evolve: Number of internal degrees of freedom: ', &
            nactive-(ntran_ic+nrota_ic)/nchromos
    endif
    call chmalloc("genetic.src","GA_evolve","where_active",nactive,intg=where_active)

    icount = 0
    Do iActive =0, nGenes/GeneSize-1
       If ( active_pointer(iActive+1) == 1 ) then
          where_active(icount+1) = iActive + 1
          icount = icount + 1
       Endif
    EndDo
    ! getting the space alocated for center of mass coordinates
    ! and euler angles
    !

    call chmalloc("genetic.src","GA_evolve","ntran",3*nchromos,crl=ntran)
    call chmalloc("genetic.src","GA_evolve","prob",nchromos,crl=prob)
    call chmalloc("genetic.src","GA_evolve","nterm",nchromos*lenent,crl=nterm)

    if(qrota) then
       call chmalloc("genetic.src","GA_evolve","nrota",3*nchromos,crl=nrota)
    endif
    !
    nGenes =  ( nActive * Genesize )

    call chmalloc("genetic.src","GA_evolve","chromos",ngenes,nchromos,crl=chromos)
    call chmalloc("genetic.src","GA_evolve","chromos_rank",nchromos,intg=chromos_rank)
    call chmalloc("genetic.src","GA_evolve","acce",nchromos,intg=acce)

    !
    !  Initialize the logical array to flag energy evaluation for population
    !  and evaluate energy
    Do iCopy = 1, nChromos
       qTemp = ThisOne(qParent, iCopy, .true.)
    Enddo

    call init_rank(chromos_rank,nchromos,nmonte,ranktemp &
         ,qparent,nparents,acce,nterm)
    !
    !   end of memory alocation
    !
    ! temperature scaling in MC procedure
    !

    !
    !  If the initialization includes randomizing the ICs, choose them randomly
    If ( qInitialize ) then
       if(qtran.or.qrota) &
            call GA_RIGID(nchromos, onAtom, is1, is2, is3, X, Y, Z, &
            nTran_IC, nrota_IC,ntran,nrota)
       Call GA_Random( nChromos, StepDihe, StepAngl, StepBond, &
            StepTran, StepRota,step_improp,idepend, &
            nBondL_IC, icr_struct%B1ic, &
            nBondR_IC, icr_struct%B2ic, &
            nAnglL_IC, icr_struct%T1ic, &
            nAnglR_IC, icr_struct%T2ic, &
            nDihe_IC, icr_struct%PIC, &
            nTran_IC,ntran, nrota_IC,nrota, &
            rdihe,rtran,rrota,offtra,xt1,yt1,zt1, active_pointer)

       !
       Call GA_Place(is1,is2,is3,nChromos, onAtom, X, Y, Z, &
            oLenIC, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            IC_Rij, IC_Rjk, IC_Tht, qParent)
       Call GA_Build(nChromos, X, Y, Z, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR, &
            oLenIC, onAtom*nChromos, qParent)
       !
       ! finding the initial CM positions and initial Eulers for
       ! all chromosomes
       !
       !
       !
       if(qtran.or.qrota) &
            Call GA_rotate(qtran,qrota,nchromos, onAtom, X, Y, Z, &
            nTran_IC,ntran, nrota_IC,nrota,qparent,1)
    ELSE
       !
       ! rotation of the replicas to the desired values
       !
       if(qtran.or.qrota) &
            call GA_RIGID(nchromos, onAtom, is1, is2, is3, X, Y, Z, &
            nTran_IC, nrota_IC,ntran,nrota)
       !
       Call GA_Place(is1,is2,is3,nChromos, onAtom, X, Y, Z, &
            icr_struct%LenIC, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            IC_Rij, IC_Rjk, IC_Tht, qParent)
       Call GA_Build(nChromos, X, Y, Z, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR, &
            oLenIC, onAtom*nChromos, qParent)
       !
       if(qtran.or.qrota) then
          Call GA_rotate(qtran,qrota,nchromos, onAtom, X, Y, Z, &
               nTran_IC,ntran, nrota_IC,nrota,qparent, &
               nmonte)
          call GA_RIGID(nchromos, onAtom, is1, is2, is3, X, Y, Z, &
               nTran_IC, nrota_IC,ntran,nrota)
       endif
       !
    ENDIF
    ievaluat=0
    Call GA_Ecntrl(GA_Energy, .true. &
         ,ievaluat,qparent,igeneration,nupdate,nterm &
         , nprint1,chromos_rank)
    ievaluat=1
    !****CLBIII the boltzmann constant is already in CHARMM in appropriate units
    kt1=0.5919459*tstart/298.1
    if(qmonte) &
         Call boltzman(ievaluat,nChromos, nNiches, &
         GA_Energy,chromos_rank, &
         nParents, qParent, &
         .false.,tstart,kt1,acce,ijwalk,kt1,nterm &
         ,natom )
    ievaluat=0
    !
    if(qmonte) then
       temp=tstart
       kt1=0.5919459*temp/298.15
    endif
    steptr1=steptran
    stepro1=steprota
    stepdih1=stepdihe
    Do While ( (iGeneration <= ngeneration) .and. .not. Converged )
       if(ievaluat <= nevaluat) then
          if(qmonte) then
             if(mod(iGeneration,ncycle) == 0 ) then
                temp=temp+tincr
                kt1=0.5919459*temp/298.15
             endif
          endif
          temp_store=temp
          !
          if(mod(igeneration,ijwalk) == 0) then
             ktb=0.5919459*tbigstep/298.15
          endif
          !
          if(mod(igeneration,ibigstep) == 0) then
             !
             steptran=bsteptr
             steprota=bstepro
             stepdihe=bstepdih
          else
             steptran=steptr1
             steprota=stepro1
             stepdihe=stepdih1
          endif
          !
          Call GA_Move( nChromos, nParents, &
               nchildren, GeneSize, &
               nNiches, nCombineNiches, iGeneration, &
               MutRate, &
               StepDihe, StepAngl, StepBond,step_improp, &
               rsame,nterm,nprint1, &
               CrosRate,Steptran,steprota,qconv,qconve,qmonte,pint1, &
               nBondL_IC, icr_struct%B1ic, &
               nBondR_IC, icr_struct%B2ic, &
               nAnglL_IC, icr_struct%T1ic, &
               nAnglR_IC, icr_struct%T2ic, &
               nDihe_IC, icr_struct%PIC, &
               ntran_IC,ntran,nrota_IC,nrota, &
               oLenic,idepend,vdepend, &
               qParent, GA_Energy, nGenes, &
               Chromos, Chromos_Rank, &
               nActive, where_active , well_dev,prob, &
               qsteady,qdepend,dpres,pall,pcall,ptall)
          !
          Call GA_Place(is1,is2,is3,nChromos, onAtom, X, Y, Z, &
               oLenIC, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               IC_Rij, IC_Rjk, IC_Tht, qParent)
          !
          Call GA_Build(nChromos, X, Y, Z, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               oLenIC, onAtom*nChromos, qParent)
          !
          i=0
          if(qtran.or.qrota) &
               Call GA_rotate(qtran,qrota,nchromos, onAtom, X, Y, Z, &
               nTran_IC,ntran, nrota_IC,nrota, &
               qparent,i)
          !
          if(mod(igeneration,nupdate) == 0) &
               call init_update(chromos_rank,nchromos,qparent)

          If ( Mod ( igeneration, iPrint )  ==  0 ) qPrntEnergy = .true.
          Call GA_Ecntrl(GA_Energy, .false.,  &
               ievaluat,qparent,igeneration,nupdate,nterm &
               ,nprint1,chromos_rank )
          !
          if(qmonte) then
             Call boltzman(igeneration,nChromos, nNiches, &
                  GA_Energy,chromos_rank, &
                  nParents, qParent, &
                  qPrntEnergy,temp,kt1,acce,ijwalk,ktb,nterm, &
                  natom )
          else

             call chmalloc("genetic.src","GA_evolve","already_ranked",nchromos, &
                  log=already_ranked)

             if(mod(igeneration,nupdate) == 0.or. &
                  mod(igeneration,nCombineNiches) == 0) then
                Call Fitsort (igeneration,nChromos, nNiches, GA_Energy, &
                     Chromos_Rank, nParents, nchildren,qParent, &
                     already_ranked,qPrntEnergy,prob,.false., &
                     dpres,nterm,nprint1)
             else
                Call Fitsort (igeneration,nChromos, nNiches, GA_Energy, &
                     Chromos_Rank, nParents, nchildren,qParent, &
                     already_ranked,qPrntEnergy,prob,qsteady, &
                     dpres,nterm,nprint1)
             endif

             call chmdealloc("genetic.src","GA_evolve","already_ranked",nchromos, &
                  log=already_ranked)

          endif
          !
          ! The following call is trnasfered from GA_Moveroutine
          !
          if(qconv.or.qconve) then
             Call Well_Converg(nNiches,nchromos, ndihe_IC, &
                  ntran_IC,nrota_IC, &
                  start_angll,start_rota, &
                  icr_struct%PIC, &
                  ntran,nrota, &
                  nActive, where_active, &
                  well_dev, &
                  nparents,chromos_rank,qconv,qconve,GA_energy)
          endif

          !       APH: this call was unneccessary

          qPrntEnergy = .false.
          if(qconv) then
             If ( (well_dev)  <=  Tolco ) then
                Converged = .true.
                if(prnlev > 1)  Write(OutU,'(a,/,a,i10,a,/,a,i10)') &
                     ' GA_Evolve: Convergence reached - internal coordinate', &
                     ' Evolved for ', igeneration,' generations', &
                     ' The total number of energy evaluation was ',ievaluat

             Endif
          endif
          !
          if(qconve) then
             If ( (well_dev)  <=  Toler ) then
                Converged = .true.
                if(prnlev > 1)  Write(OutU,'(a,/,a,i10,a,/,a,i10)') &
                     ' GA_Evolve: Convergence reached - energy', &
                     ' Evolved for ', igeneration,' generations', &
                     ' The total number of energy evaluation was ',ievaluat

             Endif
          endif

          ELast = ECurrent

          iGeneration = iGeneration + 1
          temp=temp_store

          ECurrent = 0.0
40        format('GA_ENERGY  ',i5,2x,f14.4)
       ELSE
          if(prnlev > 1)  Write(OutU,'(a,/,a,i10,a,/,a,i10)') &
               ' GA_Evolve: Maximum number of energy evaluation exceeded ', &
               ' Evolved for ', igeneration,' generations', &
               ' The total number of energy evaluation was ',ievaluat
          goto 41
       ENDIF
    Enddo

41  continue
    if(nmonte > 0.and.(qtran.or.qrota)) &
         call GA_RIGID(nchromos, onAtom, is1, is2, is3, X, Y, Z, &
         nTran_IC, nrota_IC,ntran,nrota)
    !
    If ( .not. Converged )then
       if(.not.(qconv.or.qconve)) then
          if(prnlev > 1) Write(OutU,'(a,/,a,i10,a,/,a,i10)') &
               ' GA_Evolve: Convergence not checked    ', &
               ' Evolved for ', igeneration,' generations', &
               ' The total number of energy evaluation was ',ievaluat
       else
          if(prnlev > 1) Write(OutU,'(a,/,a,i10,a,/,a,i10)') &
               ' GA_Evolve: Convergence >>NOT<< reached', &
               ' Evolved for ', igeneration,' generations', &
               ' The total number of energy evaluation was ',ievaluat
       endif
    endif
    if(prnlev > 1) write(outu,'(a,i4,a/)') &
         ' GA_Evolve : FINAL ENERGIES OF TOP',nprint1,' CHROMOSOMES'
    igeneration=-5
    Call GA_Ecntrl(GA_Energy, .true. &
         ,ievaluat,qparent,igeneration,nupdate,nterm &
         , nprint1,chromos_rank)
    !
    ! now deleting the structures which are not fit enough
    !
    if(qdelete) then
       Call GA_delunfit(nchromos,onatom*nchromos,natom, &
            nchromos, &
            chromos_rank,nniches ,nparents,nchildren,ndelete)
       if(prnlev > 1) &
            write(outu,'(a,i10,a)') '  GA_Evolve : after deletetion ', &
            nparents, ' are left'
    endif
    !  Reset energy flags to original values
    If ( nDihe_IC  <=  0 ) QETerm(Dihe) = QDihe
    If ( nDihe_IC  <=  0 ) QETerm(ImDihe) = QImDihe
    If ( ( nAnglL_IC + nAnglR_IC )  <=  0 ) QETerm(Angle) = QAngle
    If ( ( nAnglL_IC + nAnglR_IC )  <=  0 ) QETerm(UreyB) = QUreyB
    If ( ( nBondL_IC + nBondR_IC )  <=  0 ) QETerm(Bond) = QBond
    qGA_Ener = .false.
    qeterm(charm)=qchar
    qeterm(user)=quser
    !
    qmonte=.false.
    call save_rank(chromos_rank,ranktemp,nchromos)

    call chmdealloc("genetic.src","GA_evolve","active_pointer",integer1,intg=active_pointer)

    call chmdealloc("genetic.src","GA_evolve","acce",nchromos,intg=acce)

    call chmdealloc("genetic.src","GA_evolve","where_active",nactive,intg=where_active)

    call chmdealloc("genetic.src","GA_evolve","qParent",nChromos,log=qParent)
    call chmdealloc("genetic.src","GA_evolve","prob",nchromos,crl=prob)
    call chmdealloc("genetic.src","GA_evolve","nterm",nchromos*lenent,crl=nterm)
    call chmdealloc("genetic.src","GA_evolve","chromos",ngenes,nchromos,crl=chromos)
    call chmdealloc("genetic.src","GA_evolve","ntran",3*nchromos,crl=ntran)

    if(qrota) then
       call chmdealloc("genetic.src","GA_evolve","nrota",3*nchromos,crl=nrota)
    endif
    call chmdealloc("genetic.src","GA_evolve","chromos_rank",nchromos,intg=chromos_rank)

    !
    if(qclear) then
       if(ntran_IC > 0) call chmdealloc("genetic.src","GA_evolve","tran_active",3, &
            intg=tran_active)
       if(nrota_IC > 0) call chmdealloc("genetic.src","GA_evolve","rota_active",3, &
            intg=rota_active)
       If ( nDihe_IC  >  0 ) then
          call chmdealloc("genetic.src","GA_evolve","Dihe_active",olenIC,intg=Dihe_active)
       Endif

       If ( ( nAnglL_IC + nAnglR_IC )  >  0 ) then

          If (nAnglL_IC  >  0) then
             call chmdealloc("genetic.src","GA_evolve","AnglL_active",olenIC,intg=AnglL_active)
          Endif

          If (nAnglR_IC  >  0) then
             call chmdealloc("genetic.src","GA_evolve","AnglR_active",olenIC,intg=AnglR_active)
          Endif

       Endif

       If ( ( nBondL_IC + nBondR_IC )  >  0 ) then

          If (nBondL_IC  >  0) then
             call chmdealloc("genetic.src","GA_evolve","BondL_active",olenIC,intg=BondL_active)
          Endif

          If (nBondR_IC  >  0) then
             call chmdealloc("genetic.src","GA_evolve","BondR_active",olenIC,intg=BondR_active)
          Endif
       endif
       call chmdealloc("genetic.src","GA_setup","idepend",olenIC,intg=idepend)
       call chmdealloc("genetic.src","GA_setup","idepend",olenIC,crl=vdepend)

    endif
    qgalgor=.false.
#if KEY_BLOCK==1
    NOFORC=.false.  
#endif

    Return
  End Subroutine GA_evolve
  !
  Subroutine GA_delunfit(ncopy,natom,natomT, &
       nchromos, &
       rank,nniches ,nparents,nchildren,ndelete)
  use intcor_module
  use memory
  use stream
    !
    !  Local variables
    INTEGER rank(*)
    integer ncopy, natom,nstart,nstop,icopy,onatom,ii
    integer natomt,iffni,nniches,start,inich,nchromos
    integer icopy1,nparents,nchildren,ndelete
    integer,allocatable,dimension(:) :: igrplt, imap, invmap, irglst, isglst
    integer,allocatable,dimension(:) :: jslct
    !
    call chmalloc("genetic.src","GA_delunfit","imap",natomt,intg=imap)
    call chmalloc("genetic.src","GA_delunfit","invmap",natomt,intg=invmap)
    call chmalloc("genetic.src","GA_delunfit","isglst",natomt,intg=isglst)
    call chmalloc("genetic.src","GA_delunfit","irglst",natomt,intg=irglst)
    call chmalloc("genetic.src","GA_delunfit","igrplt",natomt,intg=igrplt)

    call chmalloc("genetic.src","GA_delunfit","jslct",natomt,intg=jslct)

    onatom=natom/ncopy
    ii=0
    iffni=(ndelete)/nNiches
    do inich=0, nNiches-1
       start=inich*nChromos/nNiches
       Do icopy1 = start, start+iffni-1
          icopy=rank(icopy1+1)-1
          ii=ii+1
          nStart = 1 + iCopy*onatom
          nStop = (iCopy+1)*onatom
          Call MvArray1(jslct, nstart, nstop ,natom,natomT,ii)
       end do
    end do
    !
    Call GA_DelAtm(jslct, imap, invmap, isglst, irglst, igrplt)
    !
    CALL PSFSUM(OUTU)
    call chmdealloc("genetic.src","GA_delunfit","imap",natomt,intg=imap)
    call chmdealloc("genetic.src","GA_delunfit","invmap",natomt,intg=invmap)
    call chmdealloc("genetic.src","GA_delunfit","isglst",natomt,intg=isglst)
    call chmdealloc("genetic.src","GA_delunfit","irglst",natomt,intg=irglst)
    call chmdealloc("genetic.src","GA_delunfit","igrplt",natomt,intg=igrplt)
    !
    ! . Print out the structure file counters.
    call chmdealloc("genetic.src","GA_delunfit","jslct",natomt,intg=jslct)

    Return
  End Subroutine GA_delunfit


  Subroutine GA_seed(I,J,K, LenIC, B1IC,B2IC, &
       T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR, &
       IC_Rij, IC_Rjk, IC_Tht)
    !
    !  THis routine pulls off the indices into the IC tables to seed
    !  the building of the evolving GA structures.  THis routine is
    !  essentially the routine SEED by B. Brooks with the initial
    !  placement of atoms eliminated and only the indices into the
    !  IC table extracted.
  use number
  use consta
  use stream

    INTEGER I,J,K
    INTEGER LENIC
    INTEGER IC_Rij, IC_Rjk, IC_Tht
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER IIC
    !
    IC_Rij = 0
    IC_Rjk = 0
    IC_Tht = 0
    !
    !  The values taken on by the variables IC_Rij(jk), IC_Tht and IC_Phi are
    !  the corresponding IC values with sign depending on whether they are taken
    !  from the left(positive sign) or right (negative sign) entries of the IC
    !  table

    DO IIC=1,LENIC
       IF(B2IC(IIC) > 0.0) THEN
          IF(IAR(IIC) == I.AND.JAR(IIC).EQ.J.AND..NOT.TAR(IIC)) &
               IC_RIJ=-IIC
          IF(IAR(IIC) == J.AND.JAR(IIC).EQ.I.AND..NOT.TAR(IIC)) &
               IC_RIJ=-IIC
          IF(IAR(IIC) == I.AND.KAR(IIC).EQ.J.AND.TAR(IIC)) &
               IC_RIJ=-IIC
          IF(IAR(IIC) == J.AND.KAR(IIC).EQ.I.AND.TAR(IIC)) &
               IC_RIJ=-IIC
          IF(IAR(IIC) == K.AND.JAR(IIC).EQ.J.AND..NOT.TAR(IIC)) &
               IC_RJK=-IIC
          IF(IAR(IIC) == J.AND.JAR(IIC).EQ.K.AND..NOT.TAR(IIC)) &
               IC_RJK=-IIC
          IF(IAR(IIC) == K.AND.KAR(IIC).EQ.J.AND.TAR(IIC)) &
               IC_RJK=-IIC
          IF(IAR(IIC) == J.AND.KAR(IIC).EQ.K.AND.TAR(IIC)) &
               IC_RJK=-IIC
       ENDIF
       IF(B1IC(IIC) > 0.0) THEN
          IF(LAR(IIC) == I.AND.KAR(IIC).EQ.J) IC_RIJ=IIC
          IF(LAR(IIC) == J.AND.KAR(IIC).EQ.I) IC_RIJ=IIC
          IF(LAR(IIC) == K.AND.KAR(IIC).EQ.J) IC_RJK=IIC
          IF(LAR(IIC) == J.AND.KAR(IIC).EQ.K) IC_RJK=IIC
       ENDIF
       !
       IF(KAR(IIC) == J) THEN
          IF(T1IC(IIC) > 0.0) THEN
             IF(LAR(IIC) == I.AND.JAR(IIC).EQ.K) IC_THT=IIC
             IF(LAR(IIC) == K.AND.JAR(IIC).EQ.I) IC_THT=IIC
          ENDIF
          IF(T2IC(IIC) > 0.0) THEN
             IF(IAR(IIC) == I.AND.JAR(IIC).EQ.K.AND.TAR(IIC)) &
                  IC_THT=-IIC
             IF(IAR(IIC) == K.AND.JAR(IIC).EQ.I.AND.TAR(IIC)) &
                  IC_THT=-IIC
          ENDIF
       ELSE
          IF(JAR(IIC) == J.AND..NOT.TAR(IIC).AND.T2IC(IIC) > 0.0)THEN
             IF(IAR(IIC) == I.AND.KAR(IIC).EQ.K) IC_THT=-IIC
             IF(IAR(IIC) == K.AND.KAR(IIC).EQ.I) IC_THT=-IIC
          ENDIF
       ENDIF
    ENDDO

    Return
  End Subroutine GA_seed
  !

  Subroutine GA_place(is1,is2,is3,nCopy,nAtom,X,Y,Z,LENIC, &
       B1IC,B2IC,T1IC,T2IC, &
       IC_Rij, IC_Rjk, IC_Tht, qParent)
    !
    !     THIS ROUTINE CREATES THE COORDINATES FOR THE FIRST THREE ATOMS
    !     FOR SUBSEQUENT building of the GA structures by GA_build
    !
  use number
  use consta

    INTEGER nCopy, nAtom, IC_Rij, IC_Rjk, IC_Tht,is1,is2,is3
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*)
    LOGICAL qParent(*)
    !
    real(chm_real) RIJ,RJK,THETA
    INTEGER I, J, K, iShift, iCopy, iAtom
    !
    I = is1
    J = is2
    K = is3

    Do iCopy = 0, nCopy-1

       If ( qParent(iCopy+1) ) then
          Do iAtom = 1, nAtom
             X(iAtom+icopy*natom) = ANUM
             Y(iAtom+icopy*natom) = ANUM
             Z(iAtom+icopy*natom) = ANUM
          EndDo
          RIJ=0.0
          RJK=0.0
          THETA=0.0

          iShift = iCopy*LenIC

          If ( IC_Rij  <  0 ) RIJ = B2IC(-IC_Rij+iShift)
          If ( IC_Rij  >  0 ) RIJ = B1IC(IC_Rij+iShift)
          If ( IC_Rjk  <  0 ) RJK = B2IC(-IC_Rjk+iShift)
          If ( IC_Rjk  >  0 ) RJK = B1IC(IC_Rjk+iShift)
          If ( IC_Tht  <  0 ) Theta = T2IC(-IC_Tht+iShift)
          If ( IC_Tht  >  0 ) Theta = T1IC(IC_Tht+iShift)
          !
          X(I)=0.0
          Y(I)=0.0
          Z(I)=0.0
          X(J)=RIJ
          Y(J)=0.0
          Z(J)=0.0
          THETA=THETA*(PI/180.0)
          X(K)=RIJ-RJK*COS(THETA)
          Y(K)=RJK*SIN(THETA)
          Z(K)=0.0

          !

       ENDIF
       I = I + nAtom
       J = J + nAtom
       K = K + nAtom

    Enddo
    !
    RETURN
  END Subroutine GA_place


  Subroutine GA_rotate(qtran,qrota,nCopy,natom, X, Y, Z, &
       ntran,tran, nrota,rota,qparent,num)
    !
    ! This subroutine is created by M. Vieth to
    ! rotate and translate molecule
    ! generated after GA_PLACE and GA_BUILDc based on the values of
    ! cm position and euler angles
    !
    !
  use stream

    INTEGER nCopy,num
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) rota(*),tran(*)
    INTEGER nAtom,ntran,nrota, is1, is2, is3
    LOGICAL qtran,qrota,qparent(*)
    !
    ! LOCAL VARIABLES
    !
    real(chm_real) cm(3),u(9),f,t,p,xn,yn,zn
    INTEGER II, iCopy,nstart,nstop
    INTEGER i0,i1,i2,i3,icopy3
    !
    Do iCopy = 0, nCopy-1
       nStart = 1 + iCopy*natom
       nStop = (iCopy+1)*natom
       icopy3=icopy*3
       i1=1 + iCopy3
       i2=2 + icopy3
       i3=3 + icopy3
       !
       If ( qParent(iCopy+1).or.num  == 1 ) then
          nStart = 1 + iCopy*natom
          nStop = (iCopy+1)*natom
          !
          ! center of mass first of the molecule with the first atom at the origin
          !
          !
          ! first rotation of the system to the values of euler angles
          ! given in rota() array
          !
          cm(1)=0.
          cm(2)=0.
          cm(3)=0.
25        format(' cms ',3(f12.4,2x))
26        format('first ',3(f12.4,2x))
          if(qrota) then
             f=rota(i1)
             t=rota(i2)
             p=rota(i3)
             !
22           format(' f, t, p ',i5,2x,3(f12.4,2x))
             !
             ! euler angles above
             !
             u(1)=cos(p)*cos(f)-cos(t)*sin(f)*sin(p)
             u(2)=-sin(p)*cos(f)-cos(t)*sin(f)*cos(p)
             u(3)=sin(t)*sin(f)
             !
             u(4)=cos(p)*sin(f)+cos(t)*cos(f)*sin(p)
             u(5)=-1.*sin(p)*sin(f)+cos(t)*cos(f)*cos(p)
             u(6)=-sin(t)*cos(f)
             !
             u(7)=sin(t)*sin(p)
             u(8)=sin(t)*cos(p)
             u(9)=cos(t)
             !
             do ii=nstart,nstop
                xn=x(ii)*u(1)+y(ii)*u(2)+z(ii)*u(3)
                yn=x(ii)*u(4)+y(ii)*u(5)+z(ii)*u(6)
                zn=x(ii)*u(7)+y(ii)*u(8)+z(ii)*u(9)
                x(ii)=xn
                y(ii)=yn
                z(ii)=zn
             end do
             !
          endif
          !
          ! Find atom whose coordinates are to be generated (atom l).
          !
          DO II=NSTART,NSTOP
             cm(1)=cm(1)+x(ii)
             cm(2)=cm(2)+y(ii)
             cm(3)=cm(3)+z(ii)
          end do
          !
          ! center of mass defined, computing translation vector to the new CM
          !
          i0=0
          do ii=i1,i3
             i0=i0+1
             cm(i0)=tran(ii)-cm(i0)/natom
          end do
          !
          ! translation to the cm positions specified in tran() array
          !
          DO II=NSTART,NSTOP
             x(ii)=x(ii)+cm(1)
             y(ii)=y(ii)+cm(2)
             z(ii)=z(ii)+cm(3)
          end do
          !
          ! testing the new cm
          !

          cm(1)=0.
          cm(2)=0.
          cm(3)=0.
          DO II=NSTART,NSTOP
             cm(1)=cm(1)+x(ii)
             cm(2)=cm(2)+y(ii)
             cm(3)=cm(3)+z(ii)
          END DO
       endif
20     format(1x,2i5,3(f12.3,2x))
    END DO
    return
  end Subroutine GA_rotate

  Subroutine GA_rigid(nCopy,natom, is1, is2, is3, X, Y, Z, &
       ntran,nrota,tran,rota)
    !
    ! This subroutine is created by M.Vieth to compute euler angles
    ! for each replica and their centers of masses
    !
    !
  use stream
  use consta

    INTEGER nCopy
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) rota(*),tran(*),sins,coss
    INTEGER nAtom,ntran,nrota, is1, is2, is3
    real(chm_real) x1z2,x1y3,y1x2,y1z3,y2z1,y2z3,sum,a32,a33,a23 &
         ,anorm1,anorm2,anorm3,anorm4,phi,theta,psi &
         ,x1(3),y1(3),z1(3),sint,phi1,psi1,diff,diff1
    INTEGER II, iCopy,nstart,nstop,icopy3
    INTEGER i1,i2,i3,n1,n2,n3,i
    !
    Do iCopy = 0, nCopy-1
       nStart = 1 + iCopy*natom
       nStop = (iCopy+1)*natom
       icopy3=icopy*3
       i1=1 + icopy3
       i2=2 + icopy3
       i3=3 + icopy3
       !
       ! center of mass first
       !
       tran(i1)=0.
       tran(i2)=0.
       tran(i3)=0.
       DO II=NSTART,NSTOP
          !
          ! Find atom whose coordinates are to be generated (atom l).
          !
          tran(i1)=tran(i1)+x(ii)
          tran(i2)=tran(i2)+y(ii)
          tran(i3)=tran(i3)+z(ii)
       END DO
       do ii=i1,i3
          tran(ii)=tran(ii)/natom
       end do
       ! center of mass defined, computing Eulers
       !
       if(nrota > 0) then
          n1=nstart + is1 - 1
          n2=nstart + is2 - 1
          n3=nstart + is3 - 1
20        format(4(E15.5,2x))
          x1(1)=x(n2)-x(n1) ! x of the x1
          x1(2)=y(n2)-y(n1) ! y of the x1
          x1(3)=z(n2)-z(n1) ! z of the x1
          anorm1=dsqrt(x1(1)**2+x1(2)**2+x1(3)**2)
          !
          !  anorm1 is the RIJ
          !
          do ii=1,3
             x1(ii)=x1(ii)/anorm1
          end do
          !
          ! the second vector not the y axis yet
          !
          y1(1)=x(n3)-x(n2)
          y1(2)=y(n3)-y(n2)
          y1(3)=z(n3)-z(n2)
          !
          !
          anorm2=dsqrt(y1(1)**2+y1(2)**2+y1(3)**2)
          do ii=1,3
             y1(ii)=y1(ii)/anorm2
          end do
          !
          ! anorm2 is the rjk
          !
          !
          !  the z axis is the vector product of x1 and y1
          !
          z1(1)=x1(2)*y1(3)-x1(3)*y1(2)
          z1(2)=x1(3)*y1(1)-x1(1)*y1(3)
          z1(3)=x1(1)*y1(2)-x1(2)*y1(1)
          anorm3=dsqrt(z1(1)**2+z1(2)**2+z1(3)**2)
          do ii=1,3
             z1(ii)=z1(ii)/anorm3
          end do
          if(z1(3) == 1) z1(3)=0.9999999
          !
          ! now the y axis z1Xx1
          !
          y1(1)=z1(2)*x1(3)-z1(3)*x1(2)
          y1(2)=z1(3)*x1(1)-z1(1)*x1(3)
          y1(3)=z1(1)*x1(2)-z1(2)*x1(1)
          !
21        format('axes ',3(f12.4,2x))
24        format('cms ',3(f12.4,2x))
23        format('eulers ',3(f12.4,2x))
          ! first the transformation matrix components
          !
          do ii=1,2
             theta=acos(z1(3))
             if(ii == 2)  theta=twopi-acos(z1(3))
             sint=sin(theta)
             sins=x1(3)/sint
             coss=y1(3)/sint
             if(sins > 0.and.coss.gt.0) then
                ! first quater
                psi=asin(x1(3)/sint)
                psi1=acos(y1(3)/sint)
             elseif(sins > 0.and.coss < 0) then
                ! second quater
                psi=0.5*twopi-asin(x1(3)/sint)
                psi1=acos(y1(3)/sint)
             elseif(sins < 0.and.coss.lt.0) then
                ! third quater
                psi=0.5*twopi-asin(x1(3)/sint)
                psi1=twopi-acos(y1(3)/sint)
             elseif(sins < 0.and.coss > 0) then
                ! fourth quater
                psi=twopi+asin(x1(3)/sint)
                psi1=twopi-acos(y1(3)/sint)
             endif
             !
             diff1=abs(psi1-psi)
             sins=z1(1)/sint
             coss=-z1(2)/sint
             if(sins > 0.and.coss.gt.0) then
                ! first quater
                phi=asin(z1(1)/sint)
                phi1=acos(-z1(2)/sint)
             elseif(sins > 0.and.coss < 0) then
                ! second quater
                phi=0.5*twopi-asin(z1(1)/sint)
                phi1=acos(-z1(2)/sint)
             elseif(sins < 0.and.coss.lt.0) then
                ! third quater
                phi=0.5*twopi-asin(z1(1)/sint)
                phi1=twopi-acos(-z1(2)/sint)
             elseif(sins < 0.and.coss > 0) then
                ! fourth quater
                phi=twopi+asin(z1(1)/sint)
                phi1=twopi-acos(-z1(2)/sint)
             endif
             diff=abs(phi1-phi)
             if(diff1 < 0.0001.and.diff.lt.0.0001) goto 32
          end do
          if (prnlev > 1) write(outu,*) 'big problem - no solutions ',diff1,diff
32        continue
          rota(i1)=phi
          rota(i2)=theta
          rota(i3)=psi
       endif
    END DO
    return
  end Subroutine GA_rigid

  Subroutine GA_build(nCopy, X, Y, Z, B1IC, B2IC, T1IC, T2IC, &
       PIC, IAR, JAR, KAR, LAR, TAR, LenIC, &
       nAtom, qParent)

    !  This routine builds all of the cartesian coordinates for the GA
    !  chromosomes based upon the evolving internal coordinates.  The routine
    !  is an agglomeration of BILDC and CARTCV both written by Bernie Brooks.
    !
    !     THIS ROUTINE CONSTRUCTS CARTESIAN COORDINATES FROM THE INTERNAL
    !     COORDINATES
    !
    !     By Bernard R. Brooks    1982
    !
  use stream
  use number
  use intcor2,only:cartcv
    !
    INTEGER nCopy
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    INTEGER LenIC, nAtom
    !
    real(chm_real) R,THETA,PHI
    INTEGER INC,KK,II,I,J,K,L,ITEMP, nStart, nStop, iCopy
    INTEGER CHANGE
    LOGICAL RETRY,JUNK,UNKN,OK
    LOGICAL TAR(*), qParent(*)
    !
    Do iCopy = 0, nCopy-1

       If (qParent(iCopy+1)) then

          nStart = 1 + iCopy*LenIC
          nStop = (iCopy+1)*LenIC
          INC=1
          KK=NSTART-1
          !
          CHANGE=2
          DO WHILE(CHANGE > 0)
             CHANGE=CHANGE-1
             RETRY=.FALSE.
             DO II=NSTART,NSTOP
                KK=KK+INC

                !
                ! Find atom whose coordinates are to be generated (atom l).
                !
                I=IAR(KK)
                J=JAR(KK)
                K=KAR(KK)
                L=LAR(KK)
                PHI=(PIC(KK))
                R=(B1IC(KK))
                THETA=(T1IC(KK))
                !
                IF(INC < 0) THEN
                   R=(B2IC(KK))
                   THETA=(T2IC(KK))
                   ITEMP=I
                   I=L
                   L=ITEMP
                   IF(TAR(KK)) THEN
                      PHI=-PHI
                   ELSE
                      ITEMP=J
                      J=K
                      K=ITEMP
                   ENDIF
                ENDIF
                !
                ! SEE IF COORDINATES ARE ALREADY KNOWN
                !
                IF(I <= 0) THEN
                   CONTINUE
                ELSE IF(J <= 0) THEN
                   CONTINUE
                ELSE IF(K <= 0) THEN
                   CONTINUE
                ELSE IF(L <= 0) THEN
                   CONTINUE
                ELSE IF(R < 0.001) THEN
                   CONTINUE
                ELSE IF(THETA < 0.001) THEN
                   CONTINUE
                ELSE IF(I == K) THEN
                   CONTINUE
                ELSE IF(I == J) THEN
                   CONTINUE
                ELSE IF(K == J) THEN
                   CONTINUE
                ELSE
                   !
                   ! Check to see if all antecedents are known
                   !
                   UNKN=X(L) == ANUM
                   JUNK=(X(I) == ANUM.OR.X(J).EQ.ANUM.OR.X(K).EQ.ANUM)
                   RETRY=RETRY.OR.JUNK.OR.UNKN
                   !                    RETRY=RETRY.OR.JUNK
                   IF (UNKN.AND..NOT.JUNK) THEN
                      !                    IF (.NOT.JUNK) THEN
                      !
                      ! Set geometrical parameters

                      CALL CARTCV(X,Y,Z,I,J,K,L,R,THETA,PHI,OK)
                      IF(OK) CHANGE=2
                   Endif
                   !                    endif
                ENDIF
             ENDDO
             KK=KK+INC
             INC=-INC
             !
             ! Retry will be false if all coordinates are known
          ENDDO
       Endif
    Enddo
    !
    IF(RETRY) THEN
       CALL WRNDIE(1,'<BUILDC>','SOME COORDINATES NOT BUILT')
    ELSE
       !         IF(PRNLEV >= 2) WRITE(OUTU,123)
123    FORMAT(' GA_build: ALL POSSIBLE COORDINATES HAVE BEEN PLACED')
    ENDIF
    KK=0
    DO I=1,NATOM
       IF(X(I) == ANUM) KK=KK+1
    ENDDO
    IF (KK /= 0 .AND. WRNLEV >= 2) WRITE(OUTU,124) KK
124 FORMAT(' ****  WARNING  ****',I5, &
         ' COORDINATES ARE STILL UNDEFINED')
    !
    RETURN
  END Subroutine GA_build

  Subroutine GA_Ecntrl(GA_Energy, lPrint, &
       nevaluat,par,igener,nupdate,eterm1,nprint1,rank)
    !
    !   This routine controls calls to the energy function to compute the
    !   energy of each chromosome in the GA system.  It is presently set up
    !   using the fast scalar routines and is modeled after the intere subroutine.
    !   It can, in principal, be easily parallelized since gene building and
    !   energy evaluation are uncoupled.  Alas, this will wait until a later
    !   time, pending the "usefulness" of the approach.

  use nb_module
#if KEY_RMD==1
  use cross, only: ecross,NCRUN         
#endif
#if KEY_FLUCQ==1
  use flucqm,only: fqcfor               
#endif
  use chm_types
  use dimens_fcm
  use number
  use comand
  use deriv
  use coord
  use econtmod
  use eintern
  use eintern_fast
  use mrmd_fcm,only:emrmd,mrmd_active
  use psf
  use param
  use cnst_fcm
  use code
  use inbnd
  use bases_fcm
  use energym
  use stream
  use fast
  use noem
  use block_fcm
  use grid_dock
#if KEY_FLUCQ==1
  use flucq      
#endif
  use usermod,only: usere
  use heurist,only:updeci
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif

#if KEY_CHEQ==1
    real(chm_real) ETA(1,1)                        
#endif

    INTEGER K,I,IFIRST,N,ILAST,J, iCopy,rank(*)
    INTEGER FstAtm, FstBond, FstAngl, FstPhi, FstImPhi, &
         LstAtm, LstBond, nCsPhi_GA
    INTEGER q, inoenum,nevaluat
    INTEGER igener,nupdate
    real(chm_real) EXX, GA_Energy(*), Etemp, Etemp2, Etemp3, eterm1(*) !!Yujin
    real(chm_real) rmsd_chrom, dihe_lrmsd, dihe_urmsd
    real(chm_real) temp_energy
    LOGICAL lPrint,par(*),lused1
    integer ieval,icopy1,nchromnich
    integer icopy2, jcopy,nprint1
#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       CALL WRNDIE(-5,'<GA_Ecntrl>','GA NOT READY FOR DOMDEC')
    endif
#endif /* (domdec)*/
    !  Set up loop to compute the energy contributions from each copy
    ieval=nevaluat
    lused1=.false.
    nchromnich=nchromos/nniches
    !
    ! DO the nonbonded update if necessary
    !
    if(mod(igener,nupdate) == 0) &
         CALL UPDECI(Igener,X,Y,Z,WMAIN,0,&
         (/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))

    !
    If ( lPrint.and.prnlev > 1 ) write(outu,'(a/)') &
         ' GA_Evolve : CHROMOSOME NUMBER, ENERGIES PER CHROMOSOME '
    Do jcopy = 0, nChromos -1
       !
       if(igener < 0) then
          !
          !  this is for the last printout only
          !
          icopy=rank(jcopy+1)-1
          if(par(icopy+1)) then
             par(icopy+1)=.false.
          else
             par(icopy+1)=.true.
          endif
       else
          icopy=jcopy
       endif
       !
       If( par(icopy+1) ) then

          nevaluat=nevaluat+1
          FstAtm = 1 + icopy * nAtm_chrom
          LstAtm = (icopy + 1 ) * nAtm_chrom
          FstBond = 1 + icopy * onBond
          LstBond = (icopy + 1) * onBond
          FstAngl = 1 + icopy * onTheta
          LstAngl = (icopy + 1) * onTheta
          FstPhi = 1 + icopy * onPhi
          LstPhi = (icopy + 1) * onPhi
          FstImPhi = 1 + icopy * onImPhi
          LstImPhi = (icopy+ 1) * onImPhi

          GA_Energy(icopy+1) = 0.0
          !
          DO I=1,LENENT
             ETERM(I) = ZERO
          ENDDO
          !
          DO I=FstAtm,LstAtm
             IF(IMOVE(I) <= 0) THEN
                DX(I)=ZERO
                DY(I)=ZERO
                DZ(I)=ZERO
             ENDIF
          ENDDO
          !

          IF(QETERM(USER)) THEN
             ETemp = 0.0
             CALL USERE(ETemp,X,Y,Z,DX,DY,DZ,QECONT,ECONT,LstAtm)
             ETERM(USER) = ETERM(USER) + ETemp
          ENDIF
#if KEY_RMD==1
          IF((NCRUN > 0).AND.QETERM(CROS)) THEN
             ETemp = 0.0d0
             CALL ECROSS(ETemp,X,Y,Z,DX,DY,DZ,LstAtm)
             ETERM(CROS) = ETERM(CROS) + ETemp
          ENDIF
#endif 
          IF(mrmd_active.AND.QETERM(MRMD)) THEN
             ETemp = 0.0d0
             CALL EMRMD(ETemp,X,Y,Z,DX,DY,DZ,LstAtm)
             ETERM(MRMD) = ETERM(MRMD) + ETemp
          ENDIF
          !
          !
          !     . Internal terms.
          !=======================================================================
          !     . Use the FAST option if possible.
          IF (FASTER >= 0) CALL FASTST(X,Y,Z,DX,DY,DZ,.FALSE.)
          !
          !
          !     Scalar FAST routines
          !     . Bond terms.
          IF(NBOND > 0.AND.QETERM(BOND)) THEN
             ETemp = FstBond
             CALL EBONDFS(ETemp,LstBond,IB(1),JB(1), &
                  ICB(1),CBB,CBC)
             ETERM(BOND) = ETERM(BOND) + ETemp
          ENDIF
          !
          !     . Angle terms.
          IF(NTHETA > 0.AND.QETERM(ANGLE)) THEN
             ETemp = FstAngl
             CALL EANGLFS(ETemp)
             ETERM(ANGLE) = ETERM(ANGLE) + ETemp
          ENDIF
          !
          !     . Urey-Bradley terms.
          IF(NTHETA > 0.AND.QETERM(UREYB)) THEN
             ETemp = FstAngl
             CALL EBONDFS(ETemp,LstAngl,IT(1),KT(1), &
                  ICT(1),CTUB,CTUC)
             ETERM(UREYB) = ETERM(UREYB) + ETemp
          ENDIF
          !
          !     . Proper dihedral terms.
          IF(NPHI > 0.AND.QETERM(DIHE)) THEN
             ETemp = FstPhi
             CALL EPHIFS(ETemp)
             ETERM(DIHE) = ETERM(DIHE) + ETemp
          ENDIF
          !
          !     . Improper dihedral terms.
          IF(NIMPHI > 0.AND.QETERM(IMDIHE)) THEN
             ETemp = FstImPhi
             CALL EIPHIFS(ETemp)
             ETERM(IMDIHE) = ETERM(IMDIHE) + ETemp
          ENDIF
          !
          !            Write(6,'(a,i5,a,i5)')' First atom ', fstatm,
          !     &           ' Last atom ', lstatm
#if KEY_GRID==1
          !     Grid based vdW and Elec energy terms
          !
          IF (QGrid .and. QGridOK) THEN
             !     Write(6,'(a)')' Call GridEnergy'
             ETemp = Zero
             ETemp2 = Zero
             ETemp3 = Zero
             Call GridEnergy(ETemp,ETemp2,ETemp3, &
                  XGridCenter, YGridCenter, ZGridCenter, &
                  XGridMax, YGridMax, ZGridMax, DGrid, &
                  GridForce, Cg, X, Y, Z, Dx, DY, DZ, &
                  QETERM(GrvdW), QETerm(GrElec), QETerm(GrHBON), & 
                  QGrHBon, FstAtm, LstAtm, &
                  .false., GridNoSelection)

             ETerm(GrvdW)  = ETerm(GrvdW)  + ETemp
             ETerm(GrElec) = ETerm(GrElec) + ETemp2
             ETerm(GrHBON) = ETerm(GrHBON) + ETemp3             !! Yujin Wu
          ENDIF
#endif 
          !-------------------------------------------------------------------
          !
          !=======================================================================
          !     . Nonbonded terms.
          !=======================================================================
          !
          IF(nAtm_Chrom > 0.and.(QETERM(VDW).or.QETERM(ELEC))) then
             ETemp = FstAtm
             ETemp2 = LstAtm
             CALL ENBFS8(ETemp, ETemp2, QETERM(ELEC), QETERM(VDW), 1, &
                  LstAtm, CG, bnbnd%JNB, bnbnd%INBLO, &
                  IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
                  IBLCKP,BLCOEE,BLCOEV,   & 
#endif
#if KEY_BLOCK==1
                  BLCOEVR,BLCOEVA,              & 
#endif
#if KEY_CHEQ==1
                  ETA,.FALSE.,               & 
#endif
#if KEY_FLUCQ==1
                  QFLUC,FQCFOR,              & 
#endif
#if KEY_WCA==1
                  .false.,ONE,WCA,           & 
#endif
                  lused1, .false.)

             ETERM(VDW) = ETERM(VDW) + ETemp
             ETERM(ELEC) = ETERM(ELEC) + ETemp2
          ENDIF
          !
          !=======================================================================
          !     . NOE terms.
          !=======================================================================
          !
          IF(QETERM(NOE).and.noenum > 0) then
             inoenum =  icopy+1
             etemp= inoenum
             CALL NOECNS(Etemp,DX,DY,DZ,X,Y,Z,.false.,ECONT, &
                  iNOENUM,NOESCA,NOEIPT,NOEJPT,NOEINM, &
                  NOEJNM,NOELIS,NOEEXP,NOERMN, &
                  NOEKMN,NOERMX,NOEKMX,NOEFMX, &
                  NOETCN,NOEAVE,NOEMIN,0,0,.false. &
                                !JH (soft asymptote)
                  ,NOERSW,NOESEX,NOERAM &
#if KEY_PNOE==1
                  , IsPNOE, C0X, C0Y, C0Z   & 
#endif
#if KEY_PNOE==1
                  , MVPNOE,OC0X,OC0Y,OC0Z   & 
#endif
#if KEY_PNOE==1
                  ,TC0X,TC0Y,TC0Z   & 
#endif
#if KEY_PNOE==1
                  , NMPNOE, IMPNOE          & 
#endif
                  )
             ETERM(NOE) = ETERM(NOE) + ETemp

          ENDIF
          !=======================================================================
          !     . Constraint terms
          !=======================================================================
          !
          IF(QCNSTR.AND.QETERM(CHARM)) THEN
             ETemp = FstAtm
             ! FIXME rank mismatch
             CALL ECNST1(ETemp,REFX,REFY,REFZ,KCNSTR,LstAtm, &
                  KCEXPN,xhscale,yhscale,zhscale, &
                  NUMHSETS,IHSET, &
                  X,Y,Z,DX,DY,DZ,QECONT, &
                  ECONT,0,0,.FALSE.)
             ETERM(CHARM) = ETERM(CHARM) + ETemp
          ENDIF
          !
          IF((NCSPHI > 0) .AND. QETERM(CDIHE)) THEN
             !     Here we assume that dihedral restraints were applied to all GA chromosomes
             !     after GA setup and replication.  If this is not true then an error will
             !     occur
             !     Write(OutU,'(a)') ' Computing CDIHE E'
             ETemp = icopy * ncsphi/nChromos + 1
             nCsPhi_GA = (icopy + 1) * ncsphi/nChromos
             CALL EPHI(ETemp,ICS,JCS,KCS,LCS,ICCS,nCsPhi_GA, &
                  CCSC,CCSD,CCSB, &
                  CPCOS,CPSIN, &
                  DX,DY,DZ,X,Y,Z,.TRUE.,CCSW, &
                  QECONT,ECONT,0,(/0/),(/ZERO/),(/0/),.false. &
                  )
             ETERM(CDIHE) = ETERM(CDIHE) + ETemp
          ENDIF
          !
          EXX=ZERO
          DO I=1,LENENT
             EXX=EXX+ETERM(I)
             eterm1(icopy*lenent+i)=eterm(i)
          ENDDO
          EPROP(EPOT)=EXX
          EPROP(GRMS)=0.
          GA_Energy (icopy+1) = EXX

          If ( lPrint.and.prnlev > 1 ) then
             if(igener < 0) then
                !
                !     this is for the last printout only
                !
                if(par(icopy+1)) then
                   par(icopy+1)=.false.
                else
                   par(icopy+1)=.true.
                endif
             endif
             if(prnlev <= 3) then
                do i=1, lenent
                   eterm(i)=0.
                end do
             endif
             icopy2=mod((jcopy+1),nchromnich)
             if(icopy2 == 0) icopy2=nchromnich



             if(qmonte) then
                if(icopy2 == 1) &
                     write(outu,'(/a,/,a/)') &
                     ' MONTE CARLO energies per structure ', &
                     ' Structure number is printed in the RANK field '
                qmonte=.false.
                eprop(grms)=icopy+1

                CALL PRINTE(OutU, EPROP, ETERM, 'MC_E', 'ENR', &
                     .TRUE.,icopy2, ZERO, ZERO, .TRUE.)
                qmonte=.true.
             else
                if(icopy2 == 1) &
                     write(outu,'(/a,i5/)') ' NICHE number : ', &
                     (icopy)/nchromnich+1
                if(icopy2 <= nprint1.or.igener > 0) then
                   eprop(grms)=icopy2
                   CALL PRINTE(OutU, EPROP, ETERM, 'GA_E', 'ENR', &
                        .TRUE.,icopy+1, ZERO, ZERO, .TRUE.)
                endif
             endif
          endif
       endif
    Enddo
    !
    RETURN
  END Subroutine GA_Ecntrl

  LOGICAL FUNCTION ThisOne(array, index, initialize)
    !  THis function initializes the logical array as false, and returns this
    !  as the function values (when initialize is .true.) or returns the
    !  value of the logical vatible array(index) otherwise.

    LOGICAL array(*), initialize
    integer index

    If (initialize) then
       array(index) = .true.
       ThisOne = .true.
    Else
       ThisOne = array(index)
    Endif

    Return
  End FUNCTION ThisOne


  Subroutine GA_Random( nChromos, StepDihe, StepAngl, StepBond, &
       StepTran,StepRota,step_improp,idepend, &
       nBondL_IC, B1IC, nBondR_IC, B2IC, &
       nAnglL_IC, T1IC, nAnglR_IC, T2IC, &
       nDihe_IC, PIC, &
       ntran_IC,tran, nrota_IC,rota, &
       rdihe,rtran,rrota,offtra,xt1,yt1,zt1, &
       Active )
    !
    !  This routine randomizes the values of active variables
    !
  use clcg_mod,only:random
  use reawri
  use consta

    INTEGER nChromos,  nBondL_IC, nBondR_IC, &
         nAnglL_IC, nAnglR_IC, nDihe_IC &
         ,ntran_IC, nrota_IC

    real(chm_real)  B1IC(*), B2IC(*), T1IC(*), T2IC(*), PIC(*)

    real(chm_real)  StepDihe, StepAngl, StepBond, offtra,xt1,yt1,zt1
    real(chm_real) StepTRAn, StepRota,tran(*),rota(*)

    INTEGER Active(*),idepend(*)
    real(chm_real)  step_improp

    INTEGER IC_OffSet, iChromos, IC, OffSet,it

    real(chm_real) x, rdihe,rtran,rrota,atran1,atran2,atran3
    real(chm_real) sum
    OffSet = 0
    If ( nDihe_IC  >  0 ) then
       Do iChromos = 0, nChromos-1
          IC_OffSet = iChromos*nDihe_IC/nChromos
          Do IC = 1, nDihe_IC/nChromos
10           If ( Active(IC+OffSet)  >  0 ) then
                if(rdihe > 0) then
                   x=rdihe*(random(iseed)-0.5)
                else

                   if(idepend(ic) < -1) then
                      x = PIC(IC+IC_OffSet) &
                           + ( Random(ISeed) - 0.5 ) * Step_improp
                   else
                      x = PIC(IC+IC_OffSet) &
                           + ( Random(ISeed) - 0.5 ) * StepDihe
                   endif
                endif
                If ( x  >  180.0 ) x = x - 360.0
                If ( x  <  -180.0 ) x = x + 360.0
                PIC(IC+IC_OffSet) = x
             Endif
          Enddo
       Enddo
       OffSet = OffSet + nDihe_IC/nChromos
    Endif

    If ( ( nAnglL_IC + nAnglR_IC )  >  0 ) then

       If (nAnglL_IC  >  0) then
          Do iChromos = 0, nChromos-1
             IC_OffSet = iChromos*nAnglL_IC/nChromos
             Do IC = 1, nAnglL_IC/nChromos
                If ( Active(IC+OffSet)  >  0 ) then
                   T1IC(IC+IC_OffSet) = T1IC(IC+IC_OffSet) &
                        + ( Random(ISeed) - 0.5 ) * StepAngl
                Endif
             Enddo
          Enddo
          OffSet = OffSet + nAnglL_IC/nChromos
       Endif

       If (nAnglR_IC  >  0) then
          Do iChromos = 0, nChromos-1
             IC_OffSet = iChromos*nAnglR_IC/nChromos
             Do IC = 1, nAnglR_IC/nChromos
                If ( Active(IC+OffSet)  >  0 ) then
                   T2IC(IC+IC_OffSet) = T2IC(IC+IC_OffSet) &
                        + ( Random(ISeed) - 0.5 ) * StepAngl
                Endif
             Enddo
          Enddo
          OffSet = OffSet + nAnglR_IC/nChromos
       Endif

    Endif

    If ( ( nBondL_IC + nBondR_IC )  >  0 ) then

       If (nBondL_IC  >  0) then
          Do iChromos = 0, nChromos-1
             IC_OffSet = iChromos*nBondL_IC/nChromos
             Do IC = 1, nBondL_IC/nChromos
                If ( Active(IC+OffSet)  >  0 ) then
                   B1IC(IC+IC_OffSet) = B1IC(IC+IC_OffSet) &
                        + ( Random(ISeed) - 0.5 ) * StepBond
                Endif
             Enddo
          Enddo
          OffSet = OffSet + nBondL_IC/nChromos
       Endif

       If (nBondR_IC  >  0) then
          Do iChromos = 0, nChromos-1
             IC_OffSet = iChromos*nBondR_IC/nChromos
             Do IC = 1, nBondR_IC/nChromos
                If ( Active(IC+OffSet)  >  0 ) then
                   B2IC(IC+IC_OffSet) = B2IC(IC+IC_OffSet) &
                        + ( Random(ISeed) - 0.5 ) * StepBond
                Endif
             Enddo
          Enddo
          OffSet = OffSet + nBondR_IC/nChromos
       Endif

    Endif

    If (ntran_IC  >  0) then
       Do iChromos = 0, nChromos-1
          IC_OffSet = iChromos*ntran_IC/nChromos
          Do IC = 1, ntran_IC/nChromos
             If ( Active(IC+OffSet)  >  0 ) then
                if(rtran > 0) then
                   it=mod(ic,3)
                   if(it == 0) then
100                   atran1=xt1+(random(iseed)-0.5)*2*rtran
                      atran2=yt1+(random(iseed)-0.5)*2*rtran
                      atran3=zt1+(random(iseed)-0.5)*2*rtran
                      sum=(atran1-xt1)**2+(atran2-yt1)**2+ &
                           (atran3-zt1)**2
                      if(sum < offtra**2.or.sum > rtran**2) goto 100
                      tran(IC+IC_OffSet-2)=atran1
                      tran(IC+IC_OffSet-1)=atran2
                      tran(IC+IC_OffSet)=atran3
                      !
                   endif
                else
                   tran(IC+IC_OffSet) = tran(IC+IC_OffSet) &
                        + ( Random(ISeed) - 0.5 ) * StepTran
                endif
             Endif
          Enddo
       Enddo
       OffSet = OffSet + ntran_IC/nChromos
    Endif
    If (nrota_IC  >  0) then
       Do iChromos = 0, nChromos-1
          IC_OffSet = iChromos*nrota_IC/nChromos
          Do IC = 1, nrota_IC/nChromos
             If ( Active(IC+OffSet)  >  0 ) then
                if(rrota > 0) then
                   x = &
                        ( Random(ISeed)) * rrota
                else
                   x =  rota(IC+IC_OffSet) + &
                        ( Random(ISeed)-0.5) * StepRota
                endif
                x = MOD(x+5*twopi, twoPI)
                rota(IC+IC_OffSet)=x
             Endif
          Enddo
       Enddo
    Endif

    Return
  End Subroutine GA_Random

  !
  subroutine save_rank(rank,rankt,nchromos)
    integer rank(*),nchromos,i,rankt(*)
    do i=1, nchromos
       rankt(i)=rank(i)
    end do
    return
  end subroutine save_rank
  !
  Subroutine init_update(rank,nchromos,qparent)

    integer rank(*),nchromos,i
    logical qparent(*)
    do i=1,nchromos
       qparent(i)=.true.
       rank(i)=i
    end do
    return
  end Subroutine init_update
  !
  !
  Subroutine init_rank(rank,nchromos,nmonte,rankt,qparent &
       ,nparent,acce,eterm1)
  use energym

    ! Initialization of ranking and aceptance ratio arrays
    !
    integer rank(*),nchromos,i,nmonte,rankt(*),ii,nparent
    logical qparent(*)
    integer acce(*),j
    real(chm_real) eterm1(*)
    if(nmonte == 1) then
       do i=1,nchromos
          do j=1, lenent
             eterm1((i-1)*lenent+j)=0.
          end do
          rank(i)=i
          rankt(i)=i
          acce(i)=0
          qparent(i)=.true.
       end do
    else
       do i=1,nchromos
          acce(i)=0
          rank(i)=rankt(i)
          qparent(i)=.true.
       end do
       do i=1, nparent
          ii=rank(i)
          !       qparent(ii)=.false.
       end do
    endif
    return
  end Subroutine init_rank




  Subroutine GA_Move( nChromos, nParents, &
       nchildren, GeneSize, &
       nNiches, nCombineNiches, iGeneration, &
       MutRate, &
       StepDihe, StepAngl, StepBond,step_improp, &
       rsame,eterm,nprint1, &
       Crosrate,Steptran,steprota,qconv,qconve,qmonte,pint, &
       nBondL_IC, B1IC, nBondR_IC, B2IC, &
       nAnglL_IC, T1IC, nAnglR_IC, T2IC, &
       nDihe_IC, PIC, &
       ntran_IC,tran,nrota_IC,rota, &
       olenic,idepend,vdepend, &
       qParent, GA_Energy, nGenes, &
       Chromos, Chromos_Rank, nActive, &
       ActiveIndex, total_dev,prob,qsteady,qdepend,dpres,pall, &
       pcall,ptall)
    !
    ! Passed parameters
    !
    !    nchromos -    twice the number of chromosomes in the populations
    !    nparents -    number of parents for mating
    !    nchildren -   the number of children created from parents every generation
    !    genesize -    the number of variables coding for a gene (1)
    !    nniches  -    the number of subpopulations evolving independently
    !    ncombineniches - the frequency of migration of chromosomes between niches
    !    igeneration - the current generation number
    !    mutrate     - the rate of mutationns
    !    StepDihe      maximum step size for mutation of dihedral
    !    StepAngl      maximum step size for mutation of angle
    !    StepBond      maximum step size for mutation of bond
    !    Step_improp   maximum step size for mutation of improper dihedrals
    !    rsame         the probability of cloning
    !    Crosrate      crossover rate
    !    Steptra       maximum step size for mutation of translations
    !    steprota      maximum step size for mutation of rotations
    !    qconv         logical controling whether coordinate convergence is checked
    !    qconve        logical controling whether energy convergence is checked
    !    qmonte        logical controlling if Monte Carlo scheme is on
    !    pint          probability of internal coordinate modification
    !    nBondL_IC     number of "active" bond ICs
    !    B1IC          array with vaules of bond ICs
    !    nAnglL_IC     number of "active" angle ICs
    !    T1IC          array with values of angle ICs
    !    nDihe_IC      number of "active" dihedral ICs
    !    P1C           array with values of dihedral angle ICs
    !    ntran_IC      number of active translational variables
    !    tran          values of center of masses coordinates
    !    nrota_IC      number of active euler angles
    !    rota          values of euler angles
    !    olenic        the total number of all internal coordinates
    !    idepend       the dependency pointers for dihedral angles
    !    vdepend       the dependency values for dihedral angles
    !    qparent       the logical indicating which chromosomes are parents
    !    GA_Energy     energy of chromosomes
    !    nGenes        the number of genes/variables per chromosome
    !    Chromos       the values of genes
    !    Chromos_Rank  the position in the hierarchy of a given individual
    !    nactive       the number of active genes
    !    ActiveIndex   pointer to the values of active variables
    !    total_dev     Average RMSD from the "average chromosome"
    !    prob          probability to select individual as parent
    !    qsteady       logical indicating steady state update
    !    qdepend       logical indicating dihedral angle dependencies
    !    dpres         normalized evolutionary pressure
    !    pall          Relative probability to mutate all 3 translational or
    !                  rotational degrees of freedom
    !    pcall         Relative probability to mutate all 6 rigid body
    !                  degrees of freedom
    !    ptall          Probability to mutate all internal degrees
    !                   of freedom simultaneously
    !
    !
    !   The purpose of this subroutine created by H. Daigler, M. Vieth
    !   and Charles Brooks III is to modify the values of existing genes
    !   for GA or MC schemes
    !
    !
  use chm_kinds
  use memory
    implicit none
    !
    LOGICAL qParent(*),qconv,qmonte,qdepend,qconve,qsteady
    !
    integer olenic,idepend(*)
    INTEGER nChromos, nParents, GeneSize, &
         nNiches, nCombineNiches, iGeneration,nchildren
    INTEGER ntran_IC,nrota_IC,ntran,nrota
    INTEGER Chromos_Rank(*), nGenes
    INTEGER nBondL_IC, nBondR_IC, nAnglL_IC, nAnglR_IC, &
         nDihe_IC
    INTEGER nActive, ActiveIndex(*),nprint1
    !
    real(chm_real)  total_dev,vdepend(*),crosrate
    real(chm_real)  MutRate, &
         StepDihe, StepAngl, StepBond,rsame,step_improp
    real(chm_real)  B1IC(*), B2IC(*), T1IC(*), T2IC(*), PIC(*)
    real(chm_real) StepTran, StepRota ,pint,prob(*),pall,pcall,ptall
    real(chm_real) Tran(*), Rota(*),eterm(*)
    real(chm_real)  Chromos(ngenes, *), GA_Energy(*),dpres
    !
    ! local
    logical,allocatable,dimension(:) :: already_ranked
    real(chm_real),allocatable,dimension(:) :: kold
    !
    INTEGER start_dihed, start_anglR, start_anglL, &
         start_bondL, start_bondR &
         ,start_TRan, start_Rota
    real(chm_real)  PI, MaxBond, MinAng, MinBond
    real(chm_real) mintran,maxtran

    integer i, j, which_row, which_col, where, nDihe_act
    PI = 180.0
    MaxBond = 10.0
    MinBond = .01
    MinAng = 0.01
    mintran=-120.0
    maxtran=120.0
    start_dihed = 1
    start_anglL = nDihe_IC/nChromos + 1
    start_anglR = start_anglL + nAnglL_IC/nChromos
    start_bondL = start_anglR + nAnglR_IC/nChromos
    start_bondR = start_bondL + nBondL_IC/nChromos
    start_TRAN = start_bondR + nBondR_IC/nChromos
    start_ROTA = start_TRAN + nTRAN_IC/nChromos
    ntran=ntran_IC/nchromos
    nrota=nrota_IC/nchromos
    !
    ! This will acomodate rigid body translations and rotations
    !
    If ( iGeneration <= 1 ) THEN
       Call Arr_to_Chrom( GeneSize, nGenes, nChromos, &
            Chromos, nDihe_IC, &
            PIC, nAnglL_IC, T1IC, nAnglR_IC, &
            T2IC, nBondL_IC, B1IC, nBondR_IC, &
            B2IC, &
            ntran_ic,tran, &
            nrota_ic,rota, &
            nActive, ActiveIndex)
       if(.not.qmonte) then
          i=0
          call chmalloc('genetic2.src','GA_Move','already_ranked',nChromos,log=already_ranked)
          call Fitsort(i,nChromos,nNiches,GA_Energy,Chromos_Rank,nParents, &
               nchildren,qParent,already_ranked,.false.,prob,qsteady,dpres, &
               eterm,nprint1)
       endif
    Endif
    !
    ! THE PIECE OF THE CODE ABOVE IS ACTIVE ONLY FOR THE FIRST
    ! GENERATION
    !
    !
    If ((nCombineNiches > 0).and.( MOD( iGeneration, &
         nCombineNiches) == 0)) Then
       Call niche_inter( nNiches, nParents, qParent, &
            nChromos, nGenes, Chromos, Chromos_rank)
       Call Chrom_to_Gene(GeneSize, nGenes, nChromos, &
            nDihe_IC, nAnglL_IC, nAnglR_IC, &
            nBondL_IC, nBondR_IC, &
            nTRan_IC,nrota_IC, &
            chromos, &
            PIC, T1IC, T2IC, B1IC, B2IC, &
            tran,rota, &
            PI, &
            PI, MinAng, MaxBond, MinBond, &
            maxtran,mintran,olenic,idepend,vdepend, &
            nActive, ActiveIndex,qdepend )
       if(allocated(already_ranked)) &
            call chmdealloc('genetic2.src','GA_Move','already_ranked', &
            nChromos,log=already_ranked)
       RETURN
    Endif
10  continue
    !
    call chmalloc('genetic2.src','GA_Move','kold',ngenes*genesize,crl=kold)
    call Crossover(nGenes,GeneSize,nChromos,nNiches,Chromos,nParents, &
         nchildren,qParent,Chromos_Rank,iGeneration,rsame,MutRate, &
         start_anglL,start_bondL,start_anglR,pall,pcall,ptall,crosrate, &
         start_tran,start_rota,pint,ntran,nrota,StepDihe,StepAngl,StepBond, &
         StepTRan,StepRota,step_improp,ActiveIndex,prob,qmonte,idepend, &
         kold)
    !
    Call Chrom_to_Gene(GeneSize, nGenes, nChromos, &
         nDihe_IC, nAnglL_IC, nAnglR_IC, &
         nBondL_IC, nBondR_IC, &
         nTRan_IC,nrota_IC, &
         chromos, &
         PIC, T1IC, T2IC, B1IC, B2IC, &
         tran,rota, &
         PI, &
         PI, MinAng, MaxBond, MinBond, &
         maxtran,mintran,olenic,idepend,vdepend, &
         nActive, ActiveIndex,qdepend )
    call chmdealloc('genetic2.src','GA_Move','kold', &
         ngenes*genesize,crl=kold)

    !
    Return
  End Subroutine GA_Move

  Subroutine Arr_to_Chrom(GeneSize, nGenes, nChromos, &
       chromos, nDihed_IC, Dihed_arr, &
       nAnglL_IC, AnglL_arr, &
       nAnglR_IC, AnglR_arr, &
       nBondL_IC, BondL_arr, &
       nBondR_IC, BondR_arr, &
       ntran_IC,tran_arr, &
       nrota_IC,rota_arr, &
       nActive, ActiveIndex )
    !
  use chm_kinds
    implicit none
    !
    ! description of passed parameters is present in GA_move
    !
    ! The purpose of this routine is to transfer values of internal coordinates
    ! of all chromosomes present in IC arrays into one array chromos
    INTEGER  GeneSize, nGenes, nChromos
    INTEGER  nDihed_IC, nAnglL_IC, &
         nAnglR_IC, nBondL_IC, nBondR_IC, &
         nActive, ActiveIndex(*)
    INTEGER ntran_IC,nrota_IC
    !
    real(chm_real)   chromos(nGenes, *)
    real(chm_real)   Dihed_arr(*), AnglL_arr(*), AnglR_arr(*)
    real(chm_real)   BondL_arr(*), BondR_arr(*)
    real(chm_real) rota_arr(*),tran_arr(*)
    !
    ! local
    INTEGER  gtype
    INTEGER  row, col, which_gene
    real(chm_real)   allele
    INTEGER j
    !
    !
    Do col = 1, nChromos
       Do which_gene = 1, nActive
          !
          gtype = ActiveIndex(which_gene)
          !
          If (gtype <= nDihed_IC/nChromos) Then
             allele = Dihed_arr ((col-1)*nDihed_IC/nChromos &
                  + gtype)
          Elseif (gtype <= (nDihed_IC+nAnglL_IC) &
               /nChromos) Then
             allele = AnglL_arr ((col-1)*nAnglL_IC/nChromos &
                  + gtype - nDihed_IC/nChromos)
          Elseif (gtype <= (nDihed_IC+nAnglL_IC &
               +nAnglR_IC)/nChromos) Then
             allele = AnglR_arr ((col-1)*nAnglR_IC/nChromos &
                  + gtype - (nDihed_IC+nAnglL_IC) &
                  /nChromos)
          Elseif (gtype <= (nDihed_IC+nAnglL_IC &
               +nAnglR_IC+nBondL_IC)/nChromos) Then
             allele = BondL_arr ((col-1)*nBondL_IC/nChromos &
                  + gtype - (nDihed_IC+nAnglL_IC+ &
                  nAnglR_IC)/nChromos)
             !
             ! the code below acomodates translations and rotations
             !
          Elseif (gtype <= (nDihed_IC+nAnglL_IC &
               +nAnglR_IC+nBondL_IC+nBondR_IC)/nChromos) Then
             allele = BondR_arr ((col-1)*nBondR_IC/nChromos &
                  + gtype - (nDihed_IC+nAnglL_IC+ &
                  nAnglR_IC+nBondL_IC)/nChromos)
          Elseif (gtype <= (nDihed_IC+nAnglL_IC &
               +nAnglR_IC+nBondL_IC+nBondR_IC+ntran_ic)/nChromos) Then
             !
             ! translations and rotations
             !
             allele = TRAN_arr ((col-1)*ntran_IC/nChromos &
                  + gtype - (nDihed_IC+nAnglL_IC+ &
                  nAnglR_IC+nBondL_IC+nBondR_IC)/nChromos)
          ELSEif (gtype <= (nDihed_IC+nAnglL_IC+nrota_IC &
               +nAnglR_IC+nBondL_IC+nBondR_IC+ntran_ic)/nChromos) Then
             allele = ROTA_arr ((col-1)*nrota_IC/nChromos &
                  + gtype - (nDihed_IC+nAnglL_IC+ &
                  nAnglR_IC+nBondL_IC+nBondR_IC+ntran_IC)/nChromos)
          Endif
          allele = allele/GeneSize
          Do row = 1, GeneSize
             chromos ((which_gene-1)*GeneSize + row, col) = &
                  allele
          Enddo
       Enddo
    Enddo
    Return
  End Subroutine Arr_to_Chrom

  SUBROUTINE Boltzman(igener, nChromos,nNiches, energya, &
       rank, nParents, qparents, qprint,temp,kt,acce,ibig,ktb, &
       eterm1,natom)
    !
  use chm_kinds
  use reawri
  use number
  use stream
  use energym
  use dimens_fcm
  use clcg_mod,only:random
  use deriv
    implicit none
    !
    ! The purpose of this routine is to perform Metropoilis scheme
    ! and create a new population of replicas
    !
    LOGICAL  qparents(*),qprint
    INTEGER  nChromos, nNiches, nParents,igener
    INTEGER  rank(*),acce(*),ibig
    real(chm_real) :: energya(*),temp,de,kt,enenew,k1,ktb,kt1, &
         eterm1(*)
    !
    ! local
    !
    logical qacce
    INTEGER  lowest, i, j, niche, first,jchild,ipar,i1,natom
    real(chm_real) emin,eoldg
    ! begin
    !
    !  initialize all as not having been ranked and not as parents
    Do j = 1, nChromos
       qparents (j) = .True.
    Enddo
    qacce=.false.

    if(igener >= ibig.and.mod(igener,ibig) < 4)then
       qacce=.true.
       emin=energya(rank(1))
       do j=2, nparents
          i=rank(j)
          if(energya(i) < emin) emin=energya(i)
       end do
       emin=emin+10.0
    endif
    !
    !  rank chromosomes in each niche
    !
    Do niche = 0, (nNiches-1)
       if(qprint.and.prnlev > 2) &
            write(outu,'(/a,i5,a,f8.2/)') &
            ' MONTE CARLO : GENERATION NUMBER ', &
            igener,' TEMPERATURE ',temp
99     format(' MONTE CARLO : GENERATION # TEMPERATURE ',i5,2x,f12.2/)
100    format('MONTE CARLO : ',1x,i5,1x,i5,1x,5(f12.4,1x))
       first = niche*(nChromos/nNiches)+1
       Do i = first, &
            ((niche+1)*nparents/nNiches)
          !
          ! begining of the Metropolis scheme
          !
          ipar=rank(i)
          jchild=rank(i+nparents/nniches)
          enenew=energya(jchild)
          eoldg=energya(ipar)
          de=energya(jchild)-energya(ipar)
          kt1=kt
          if(qacce) then
             if(float(acce(i))/ibig < 0.02) then
                kt1=ktb
             endif
          endif
          if(de < 0 ) then
             !
             ! mutation accepted
             !
             rank(i)=jchild
             rank(i+nparents/nniches)=ipar
             qparents(jchild)=.false.
             acce(i)=acce(i)+1
          else
             !
             ! metropolis
             !
             de=de/kt1
             if(de < 80) then
                k1=random(iseed)
                if(dexp(-de) > k1) then
                   acce(i)=acce(i)+1
                   rank(i)=jchild
                   rank(i+nparents/nniches)=ipar
                   qparents(jchild)=.false.
                else
                   qparents(ipar)=.false.
                endif
             else
                qparents(ipar)=.false.
             endif
          endif
          if(qprint.and.prnlev > 2) then
             eprop(epot)=energya(rank(i))
             eprop(grms)=acce(i)/float(igener)
             if(prnlev > 3) then
                do i1=1, lenent
                   eterm(i1)=eterm1((rank(i)-1)*lenent+i1)
                end do
             else
                do i1=1, lenent
                   eterm(i1)=0.0
                end do
             endif
             !
             call printe(outu, EPROP, eterm, 'MC_E', 'ENR', &
                  .TRUE.,i, ZERO, ZERO, .TRUE.)
          endif
       end do
    end do
    if(mod(igener,ibig) == 4) then
       do i=1, nparents
          acce(i)=0
       end do
    endif
    !
    Return
  End SUBROUTINE Boltzman

  SUBROUTINE Fitsort(igener, nChromos, nNiches, fitarr, &
       rankarr, nParents,nchildren, &
       lParentarr, already_ranked,qprint,prob1,qsteady,dpres, &
       eterm1,nprint)
    !
  use chm_kinds
  use number
  use stream
  use energym
    implicit none
    INTEGER  nChromos, nNiches, nParents,nchildren,nprint
    INTEGER  rankarr(*)
    real(chm_real)   fitarr(*)
    !
    INTEGER  lowest, i, j, niche, first,igener,last,i1
    LOGICAL  lParentarr(*)
    LOGICAL  already_ranked(*)
    logical qprint,qsteady
    integer num,ii,nparnich,ichrom,first1,isteady,last1,icount
    real(chm_real) prob1(*),sum,sum1,emax,dpres,diff,eterm1(*)
    !
    !  initialize all as not having been ranked and not as parents
    Do j = 1, nChromos
       already_ranked (j) = .False.
       lParentarr (j) = .false.
    Enddo
    !
    !  isteady is the number of steady chromosomes in the population
    !
    isteady=(nparents-nchildren)/Nniches
    !
    !  rank chromosomes in each niche
    Do niche = 0, (nNiches-1)
99     format(' GA EVOLUTION : NICHE NUMBER : ',i5/)
100    format(2x,i5,4x,i5,4x,f14.4)
       if(qsteady.and.igener > 0) then
          !
          ! Ranking for GA with steady state update
          !
          first = niche*(nChromos/nNiches)+1
          first1=first
          last=first + (nparents)/nniches-1
          last1=first +(nparents+nchildren)/Nniches -1
          !
          ! Do Not rank lowest nchildren of the population
          !
          do i=first+isteady,last
             ichrom=rankarr(i)
             rankarr(i+nchildren/nniches)=ichrom
             already_ranked(ichrom)=.true.
             !        lParentarr(ichrom)=.true.
          end do
          !
          lowest = first
          Do i = first,last
             emax=rbigst
             Do j=first,last1
                If (.NOT.already_ranked(j).and. &
                     (fitarr(j) <= emax)) Then
                   lowest =  j
                   emax=fitarr(j)
                Endif
             Enddo
             rankarr(i) = lowest
             already_ranked(lowest) = .TRUE.
          Enddo
          !
          ! end of the ranking for the steady state update
          !
       else
          !
          ! Evolutionary strategy
          !
          first = niche*(nChromos/nNiches)+1
          last=first+(nparents+nchildren)/nniches-1
          lowest = first
          Do i = niche*(nChromos/nNiches)+1,last
             emax=rbigst
             Do j=first,last
                If (.NOT.already_ranked(j).and. &
                     (fitarr(j) <= emax)) Then
                   lowest =  j
                   emax=fitarr(j)
                   !                    goto 30
                Endif
             Enddo
30           rankarr(i) = lowest
             already_ranked(lowest) = .TRUE.
          Enddo
       endif
    Enddo
    !
    !  Ranking done now
    !  mark best nParents as such
    !
    if(qprint.and.prnlev > 2) &
         write(outu,'(/a,i6)') ' GA EVOLVE : GENERATION ',igener
    nparnich=nparents/nNiches
    Do niche = 0, nNiches-1
       if(qprint.and.prnlev > 2) write(outu,99)niche+1
       first=niche*nChromos/nNiches+1
       last=(niche*nChromos+nParents)/nNiches
       !
       ! set the true values for the children part
       !
       do j=last+1,last+nchildren/nniches
          ichrom=rankarr(j)
          lParentarr(ichrom) = .true.
       end do
       icount=0
       Do j = first,last
          icount=icount+1
          ichrom=rankarr(j)
          !              lParentarr(ichrom) = .False.
          !
          ! Determining the roulette wheel selection probabilities
          !
          !        prob1(j)=(emax-fitarr(ichrom))
          !        sum=sum+prob1(j)
          !        if(qprint.and.prnlev > 2)then
          !        write(outu,100)  j,ichrom,fitarr(ichrom)
          if(qprint.and.prnlev > 2) then
             if(icount <=  nprint) then
                eprop(epot)=fitarr(ichrom)
                eprop(grms)=icount
                if(prnlev > 3) then
                   do i1=1, lenent
                      eterm(i1)=eterm1((ichrom-1)*lenent+i1)
                   end do
                else
                   do i1=1, lenent
                      eterm(i1)=0.0
                   end do
                endif
                !
                call printe(outu, EPROP, eterm, 'GA_E', 'ENR', &
                     .TRUE.,ichrom, ZERO, ZERO, .TRUE.)
             endif
             !         endif
          endif
       Enddo
       !
       ! determining the total probability as the sum of normalized
       ! scores based on the ranking and the energy
       ! Evolutionary presure of 1.1 based of scaled fitness
       ! following Jones, JMB 245, 43-53
       !
       if(igener < 1) then
          sum1=50*nparnich**2+0.5*dpres*nparnich*(nparnich-1)
          prob1(first)=(50.*nparnich+(nparnich-1)*dpres)/sum1
          j=first
          ii=1
          do j=first+1,last
             ii=ii+1
             num=50*nparnich+(nparnich-ii)*dpres
             prob1(j)=prob1(j-1)+num/sum1
          end do
          diff=1.0-prob1(last)
          do j=first,last
             prob1(j)=prob1(j)+diff
          end do
       endif
    Enddo
    !
    Return
  End SUBROUTINE Fitsort

  Subroutine Crossover ( nGenes, GeneSize, nChromos, nNiches, &
       chromos, nParents, nchildren,lparents, &
       rank, gen, rsame,mutrate, start_anglL, &
       start_bondL, start_anglR, &
       pall,pcall,ptall, Crosrate, Start_tran, Start_rota,pint, &
       ntran,nrota, &
       StepDihe, &
       StepAngl, StepBond, &
       StepTran, StepRota, &
       Step_improp, &
       ActiveIndex,prob,qmonte,idepend,kold)
    !
  use clcg_mod,only:random
  use chm_kinds
  use reawri
    implicit none
    INTEGER  nGenes, GeneSize, nChromos, nNiches
    real(chm_real)   Chromos(nGenes, *)
    INTEGER  nParents, gen
    INTEGER  rank(*),idepend(*)
    real(chm_real)   mutrate,crosrate,rsame,step_improp, &
         pall,pcall,ptall
    INTEGER  start_anglL, start_bondL, start_anglR
    INTEGER  Start_tran, Start_rota
    real(chm_real)   StepDihe, StepAngl, StepBond,pint
    real(chm_real)   StepTran, StepRota
    INTEGER  step_on, Step_off
    INTEGER  ActiveIndex(*)
    !
    INTEGER  i, j, ninte,nrigid, ntran,nrota
    real(chm_real)   k,k1
    INTEGER  chrom1, chrom2, xover, niche, next_child, igene
    real(chm_real)   kold(*)
    LOGICAL  lparents(*),qmonte,qall,qcall,qtall
    LOGICAL  next_exists,qinte
    integer jend,nchildren,j0,j1,i1,ngenes1
    real(chm_real) prob(*),k3,k12,k13,step,k14,k15,k16
    !
    ! This is the key routine to perform mating. The parents are selected
    ! based on roulette wheel selection and produce children by mutation
    ! crossover or cloning
    !
    ! begin
    Do niche = 0,(nNiches-1)
       j = ((niche*nChromos/nNiches)+nParents/nNiches+1)
       j0=niche*(nChromos/nNiches)+1
       j1=niche*(nChromos/nNiches)+nParents/nNiches
       jend=j+nchildren/nniches-1
       Do While (j <= jend)
          !
          !  otherwise, find the next non-parent chromosome.  If no more exist,
          !  a second child of the parents is not produced.
          !
          ! determination on the crossover point is based on the probability
          ! given by Pint (probability to modify the internal degress of freedom)
          ! vs. and 1-Pint (probability to modify rigid body coordinates)
          k = random(iseed)
          if(k < crosrate) then
             !
             ! Crossover
             !
             next_child = j + 1
             next_exists = .TRUE.
             If (next_child > jend &
                  ) Then
                next_exists = .False.
             Endif
             !
             ! rullete wheel selection can be used if desired
             !
             k3=random(iseed)
             do i=j0,j1
                if(k3 < prob(i)) then
                   chrom1=i
                   goto 33
                endif
             end do
33           k3=random(iseed)
             do i=j0,j1
                if(k3 < prob(i)) then
                   chrom2=i
                   goto 34
                endif
             end do
34           xover =  1 + (random(iseed))*(nGenes-1)
             !
             !     xover only no mutation performed with it
             !
             ! replacing the genes up to the crossover point
             ! from chrom1 to j-th chromosome
             !
             Do i = 1, xover
                chromos(i, rank(j)) = chromos (i,rank(chrom1))
                If ( next_exists ) Then
                   chromos(i, rank(next_child)) = &
                        chromos(i,rank(chrom2))
                Endif
             Enddo
             Do i = (xover+1), nGenes
                chromos(i, rank(j)) = chromos(i, rank(chrom2))
                IF ( next_exists ) Then
                   chromos(i,rank(next_child)) = &
                        chromos(i,rank(chrom1))
                Endif
             Enddo
             !
             j = j+2
          elseif(k < mutrate) then
             !
             !     mutation
             qall=.false.
             qtall=.false.
             qcall=.false.
             qinte=.false.
             ninte=ngenes-ntran-nrota
             if((ntran+nrota) > 0) then
                k1=random(iseed)
                nrigid=ngenes-ninte
                if(k1 < pint) then
                   !
                   ! internal coordinate modification
                   !
                   xover = 1+(random(iseed))*(ninte)
                   if(ptall-pall > 0) then
                      if(ptall-pall > random(iseed)) then
                         xover=1
                         qinte=.true.
                         qtall=.true.
                      endif
                   endif
                else
                   !
                   ! rigid body coordinate modification
                   !
                   k13=random(iseed)
                   if(k13 < pcall) then
                      !
                      ! collective motion (translations and rotations)
                      !
                      qcall=.true.
                      xover=ninte+1
                   elseif(k13 < pall) then
                      !
                      ! all 3 translational or rotational components are mutated
                      !
                      qall=.true.
                      if(random(iseed)*(ntran+nrota) < ntran) then
                         !
                         ! all translations
                         !
                         xover= ninte+1
                         step=steptran
                      else
                         step=steprota
                         xover = ninte+ntran+1
                      endif
                   else
                      !
                      !  only one translational/rotational component is mutated
                      !
                      xover = ninte+1+ (random(iseed))*(nrigid)
                   endif
                endif
             else
                !
                !  internal coordinate, because there is no rigid body degrees
                !
                k13=random(iseed)
                if(k13 > ptall) then
                   qinte=.true.
                   qtall=.true.
                   xover=1
                else
                   xover =  1 + (random(iseed))*(nGenes)
                   k13=random(iseed)
                endif
             endif
             !
             ! xover point (gene to mutate chosen) perform the mutation
             !
             if(qmonte)then
                chrom1=j
             else
                !
                ! roulette wheel selection
                !
                k3=random(iseed)
                do i=j0,j1
                   if(k3 < prob(i)) then
                      chrom1=i
                      goto 35
                   endif
                end do
             endif
35           k1=chromos(xover, rank(chrom1))
             if(qmonte) k1=chromos(xover, rank(chrom1-nparents/nniches))
             i=xover
             if(qcall) then
                if(qmonte) then
                   k12=chromos(xover+1, rank(chrom1-nparents/nniches))
                   k13=chromos(xover+2, rank(chrom1-nparents/nniches))
                   k14=chromos(xover+3, rank(chrom1-nparents/nniches))
                   k15=chromos(xover+4, rank(chrom1-nparents/nniches))
                   k16=chromos(xover+5, rank(chrom1-nparents/nniches))
                else
                   k12=chromos(xover+1, rank(chrom1))
                   k13=chromos(xover+2, rank(chrom1))
                   k14=chromos(xover+3, rank(chrom1))
                   k15=chromos(xover+4, rank(chrom1))
                   k16=chromos(xover+5, rank(chrom1))
                endif
                step=steptran
                chromos(xover,rank(j))=k1+step*(random(iseed)-0.5)
                chromos(xover+1,rank(j))=k12+step*(random(iseed)-0.5)
                chromos(xover+2,rank(j))=k13+step*(random(iseed)-0.5)
                step=steprota
                chromos(xover+3,rank(j))=k14+step*(random(iseed)-0.5)
                chromos(xover+4,rank(j))=k15+step*(random(iseed)-0.5)
                chromos(xover+5,rank(j))=k16+step*(random(iseed)-0.5)
             elseif(qall) then
                if(qmonte) then
                   k12=chromos(xover+1, rank(chrom1-nparents/nniches))
                   k13=chromos(xover+2, rank(chrom1-nparents/nniches))
                else
                   k12=chromos(xover+1, rank(chrom1))
                   k13=chromos(xover+2, rank(chrom1))
                endif
                !
                ! mutating translational degrees of freedom
                !
                chromos(xover,rank(j))=k1+step*(random(iseed)-0.5)
                chromos(xover+1,rank(j))=k12+step*(random(iseed)-0.5)
                chromos(xover+2,rank(j))=k13+step*(random(iseed)-0.5)
             elseif(qtall) then
                !                ngenes1=ngenes
                ngenes1=ninte
                if(qmonte) then
                   do i1=1, ngenes1
                      kold(i1)=chromos(i1,rank(chrom1-nparents/nniches))
                   end do
                else
                   do i1=1, ngenes1
                      kold(i1)=chromos(i1,rank(chrom1))
                   end do
                endif
                do i1=1, ngenes1
                   if(i1 < start_anglL) then
                      step = StepDihe
                      if(idepend(i1) < -1) step=step_improp
                   Elseif ( i1 < start_bondL ) Then
                      step = StepAngl
                   Elseif (i1 < start_tran) Then
                      step = StepBond
                   Elseif (i1 < start_rota) Then
                      step=StepTran
                   else
                      step=StepRota
                   endif
                   chromos(i1,rank(j))=kold(i1)+step*(random(iseed)-0.5)
                enddo
             else
                igene = ActiveIndex((i-1)/GeneSize+1)
                !
                ! call old mutation routine replacing the second ranked half of the population
                !
                chromos(i, rank(j)) = &
                     Mutation (start_anglL, start_bondL, &
                     start_anglR, &
                     Start_Tran, Start_Rota, &
                     StepDihe, &
                     StepAngl, StepBond, &
                     StepTran,StepRota,step_improp, &
                     idepend, &
                     k1, &
                     ActiveIndex((i-1)/GeneSize+1), &
                     gen)
             endif
             j=j+1
          elseif(k < rsame) then
             !
             !  one of the parents becomes child
             !
             k3=random(iseed)
             loop36: do i=j0,j1
                if(k3 < prob(i)) then
                   chrom1=i
                   exit loop36
                endif
             end do loop36

             do i=1, ngenes
                chromos(i, rank(j)) = chromos(i, rank(chrom1))
             end do
             j=j+1
          else
             j=j+1
          endif
       Enddo
    Enddo
    !
    Return
  End Subroutine Crossover
  !
  Function Mutation (start_anglL, start_bondL,  &
       start_anglR, &
       Start_TRan, Start_Rota, &
       StepDihe, StepAngl, StepBond, &
       StepTran, StepRota,step_improp, &
       idepend, &
       current_value, &
       which_gene, gen) result(mut_result)
    !
    !  This function is the modified version of Mutate and contains mutations
    !  of the first six elements of the chromosome - translations and
    !  rotations
    !
  use clcg_mod,only:random
  use chm_kinds
  use reawri
    implicit none
    !
    real(chm_real) :: mut_result
    INTEGER start_anglL, start_bondL, start_anglR
    INTEGER  start_Tran, Start_Rota
    real(chm_real)  StepDihe, StepAngl, StepBond
    real(chm_real)  StepRota, StepTran,step_improp

    INTEGER idepend(*)
    real(chm_real)  current_value
    INTEGER which_gene, gen
    !
    real(chm_real)  divisor, step
    !
    Logical lrota
    !
    !  determine what type of gene it is and find step size
    lrota = .false.
    If ( which_gene < start_anglL ) Then
       step = StepDihe
       if(idepend(which_gene) < -1) step=step_improp
    Elseif ( which_gene < start_bondL ) Then
       step = StepAngl
    Elseif (which_gene < start_tran) Then
       step = StepBond
    Elseif (which_gene < start_Rota) Then
       step =  StepTran
       !
       ! translations treated like any other mutation
       !
    Else
       !
       ! rotations treated differently
       !
       step = StepRota
       lrota=.true.
    Endif
    !  calculate the mutated value
    !
25  format('MUTATION ',4(f12.4,2x))
    mut_result = current_value + (step* &
         (random(iseed) - 0.5))
    Return
  End Function Mutation

  SUBROUTINE niche_inter(nNiches, nParents, lParentarr, &
       nChromos, nGenes, Chromos, rank)
    !
  use chm_kinds
  use reawri
    implicit none
    INTEGER  nNiches, nParents
    INTEGER  nChromos, nGenes, rank(*),nich
    real(chm_real)   Chromos (nGenes, *)
    real(chm_real) etemp
    INTEGER  row, allele, i, j, niche, worst_parent, niche_best
    LOGICAL  lParentarr(*)
    !
    ! This subroutine replace least fit individuals from each subpopulation
    ! by  the top individuals from other subpopulations
    !
    ! begin
    do i=1, nchromos
       lParentarr(i)=.false.
    end do
    Do niche = 1, nNiches
       Do j = 0, nNiches - 1
          worst_parent = rank( ((niche - 1) * nChromos + nParents) &
               /Nniches-j)
          niche_best = rank ( j * nChromos / nNiches + 1 )
          lParentarr(worst_parent)=.true.
          Do allele = 1, nGenes
             Chromos (allele, worst_parent) = &
                  Chromos (allele, niche_best)
          Enddo
       Enddo
    Enddo
    Return
  End SUBROUTINE niche_inter

  Subroutine Chrom_to_Gene(GeneSize, nGenes, nChromos, &
       nDihe_IC, nAnglL_IC, nAnglR_IC, &
       nBondL_IC, nBondR_IC, &
       ntran_IC, nrota_IC &
       , chromos, &
       dihe_arr, anglL_arr, anglR_arr, &
       bondL_arr, bondR_arr , &
       tran_arr, rota_arr &
       , PIval, max_angle, &
       min_angle, max_bond, min_bond, &
       max_tran,min_tran,olenic,idepend,vdepend, &
       nActive, ActiveIndex,qdepend )
    !
  use chm_kinds
  use consta
    implicit none
    !
    INTEGER  GeneSize, nGenes, nChromos, &
         nDihe_IC, nAnglL_IC, nAnglR_IC, &
         nBondL_IC, nBondR_IC
    !
    real(chm_real)   chromos(nGenes, *), &
         dihe_arr(*), anglL_arr(*), &
         anglR_arr(*), bondL_arr(*), &
         bondR_arr(*), PIval, &
         max_angle, min_angle, &
         max_bond, min_bond &
         , max_tran,min_tran,vdepend(*)
    INTEGER ntran_IC, nrota_IC,idepend(*)
    real(chm_real) tran_arr(*),rota_arr(*)
    INTEGER  nActive, ActiveIndex(*)
    INTEGER  row, which_chrom, which_gene, allele, i,col
    real(chm_real)   sum, diff, max, min, this_chrom
    LOGICAL  adjust, dihedral, anglL, anglR, bondL, bondR
    LOGICAL tran,rota,qdepend
    integer ii,i0,olenic,offset
    !
    ! This routine transfers information from chromos array to IC arrays
    !
    ! begin
    Do which_chrom = 1, nChromos
       !
       ! Adjusting the values of dihedral angles if dependencies are present
       !
       if(qdepend) then
          offset=(which_chrom-1)*olenic
          i0=0
          do ii=offset+1,offset+olenic
             i0=i0+1
             i=idepend(i0)
             if(i > 0) then
                i=i+offset
                dihe_arr(ii)=dihe_arr(i)+vdepend(i0)
             endif
          end do
       endif
       !
       DO which_gene = 1, nActive
          dihedral = .false.
          anglL = .false.
          anglR = .false.
          bondL = .false.
          bondR = .false.
          tran  = .false.
          rota  = .false.
          sum = 0
          adjust = .FALSE.
          Do allele = ((which_gene-1)*GeneSize)+1, &
               (which_gene*GeneSize)
             sum = sum + chromos (allele, which_chrom)
          Enddo
          !  check value against boundries-change if exceeded
          !  Values are assigned the corresponding negative angle
          !
          !
          If ( ActiveIndex(which_gene)  <=  &
               nDihe_IC / nChromos ) Then
             min = -1*PIval
             max = PIval
             dihedral = .true.
          Elseif ( ActiveIndex(which_gene)  <=  &
               ((nDihe_IC + nAnglL_IC + &
               nAnglR_IC) / nChromos) ) Then
             min = min_angle
             max = max_angle
             If ( ActiveIndex(which_gene) >  &
                  (nDihe_IC + nAnglL_IC)/nChromos)  Then
                anglR = .true.
             Else
                anglL = .true.
             Endif
          Elseif ( ActiveIndex(which_gene)  <=  &
               ((nDihe_IC + nAnglL_IC + &
               nAnglR_IC+nbondR_IC+nbondL_IC) / nChromos) ) Then
             min = min_bond
             max = max_bond
             If (ActiveIndex(which_gene) > (nDihe_IC &
                  +nAnglL_IC+nAnglR_IC+nBondL_IC) &
                  /nChromos) Then
                bondR = .true.
             ELSE
                bondL = .true.
             Endif
          ELSEIF ( ActiveIndex(which_gene)  <=  &
               ((nDihe_IC + nAnglL_IC + &
               nAnglR_IC+ nbondR_IC+nbondL_IC &
               +ntran_IC  ) / nChromos) ) Then
             min = min_tran
             max = max_tran
             tran = .true.
          ELSE
             min = min_angle
             max = twopi
             rota = .true.
          Endif
          !
          If ( sum < min ) Then
             adjust = .True.
             If ( dihedral ) Then
                sum=sum+360
             Elseif (anglR.or.anglL) Then
                sum = min_Angle
             Elseif (bondR.or.bondL) Then
                sum = min_Bond
             Elseif (tran) Then
                sum = min_tran
             ELSeif(rota) then
                sum = MOD(sum+twopi, twopi)
             Endif
             !
          Elseif (sum > max) Then
             adjust = .True.
             If ( dihedral ) Then
                sum=sum-360.0
             Elseif ( anglR.or.anglL ) Then
                sum = max_angle
             Elseif (bondR.or.bondL) Then
                sum = max_bond
             ELSEif (tran) then
                sum = max_tran
             ELSEif(rota) then
                !                  i =  (-1)**(INT(ABS(sum/twoPI)))
                sum = MOD(sum+twopi, twoPI)
             Endif
          Endif
          !
          If ( dihedral ) Then
             dihe_arr(ActiveIndex(which_gene) + &
                  (which_chrom-1)*nDihe_IC/nChromos) &
                  = sum
          Elseif ( anglL ) Then
             anglL_arr((ActiveIndex(which_gene) &
                  - nDihe_IC / nChromos) + &
                  (which_chrom-1) * nAnglL_IC/nChromos) &
                  = sum
          Elseif ( anglR ) Then
             anglR_arr((ActiveIndex(which_gene) &
                  - (nDihe_IC + nAnglL_IC) / &
                  nChromos ) + (which_chrom-1) &
                  * nAnglR_IC / nChromos ) = sum
          Elseif ( bondL ) Then
             bondL_arr( (ActiveIndex(which_gene)- &
                  (nDihe_IC+nAnglL_IC+nAnglR_IC &
                  ) / nChromos) + &
                  (which_chrom-1) * nBondL_IC/ nChromos) &
                  = sum
          Elseif ( bondR ) Then
             bondR_arr( (ActiveIndex(which_gene)- &
                  (nDihe_IC+nAnglL_IC+nAnglR_IC &
                  +nbondL_IC) &
                  / nChromos) + (which_chrom-1) &
                  * nBondR_IC / nChromos ) = sum
          Elseif ( tran ) Then
             tran_arr( (ActiveIndex(which_gene)- &
                  (nDihe_IC+nAnglL_IC+nAnglR_IC &
                  +nbondL_IC+nbondR_IC) &
                  / nChromos) + (which_chrom-1) &
                  * ntran_IC / nChromos ) = sum
          ELse
             rota_arr( (ActiveIndex(which_gene)- &
                  (nDihe_IC+nAnglL_IC+nAnglR_IC+ &
                  nbondL_IC+nbondR_IC+ntran_IC) &
                  / nChromos) + (which_chrom-1) &
                  * nrota_IC / nChromos ) = sum
          Endif
          !
          If ( adjust ) Then
             diff = sum/GeneSize
             Do row = ((which_gene-1)*GeneSize)+1, &
                  (which_gene*GeneSize)
                chromos(row, which_chrom) = diff
             Enddo
          Endif
       Enddo
       col=which_chrom
23     format(2x,i5,2x,10(f12.4,2x))
    Enddo
    Return
  End Subroutine Chrom_to_Gene

  !
  !  It may be possible to generalize this code and call it five times
  !  from the main routine.  Beginning and end points for stepping
  !  through the array would be needed, etc.
  !
  !  To determine convergence on solely the parent population, this
  !  subroutine would need to be called prior to arr_to_chrom
  !  so that the energies are updated.  An additional loop would
  !  also have to be added to go through the individual niches.
  !
  ! This subroutine is a completly new version created by M. Vieth
  !
  Subroutine Well_Converg(nNiches,nchromos,ndihe_IC, &
       ntran_IC, nrota_IC, &
       ndihe,ntran, &
       dihe_arr, &
       tran_arr, rota_arr, &
       nActive, &
       ActiveIndex, &
       total_dev, &
       nparents,rank,qconv,qconve,energy)
  use chm_kinds
  use stream
    implicit none
    LOGICAL qconv,qconve
    INTEGER  GeneSize, nGenes, &
         nDihe_IC &
         ,ntran_IC, nrota_IC,nparents,nchromos
    real(chm_real)    dihe_arr(*) &
         ,tran_arr(*), rota_arr(*),energy(*)
    !
    INTEGER  nActive, ActiveIndex(*)
    real(chm_real)   total_dev, sum2
    integer igene,ipar,ichr,offset,iactive
    integer ndihe,ntran,start,offni,nNiches,inich
    integer rank(*),offtra,offrota
    real(chm_real)   sum,diff
    !
    ! This routine checks for the convergence of the entire population
    ! based on energy or coordinate values
    !
    ! begin
    !
    offni=nparents/nNiches
    total_dev=0.
    if(qconv) then
       offtra=ndihe-1
       offrota=ntran-1
       do igene=1, nactive
          sum=0.
          sum2=0.
          iactive=ActiveIndex(igene)
          if(iactive < ndihe) then
             do inich=0, nNiches-1
                start=inich*nChromos/nNiches
                do ipar=start+1,start+offni
                   ichr=rank(ipar)
                   offset=(ichr-1)*ndihe_IC/nchromos
                   sum=sum+dihe_arr(iactive+offset)
                   sum2=sum2+dihe_arr(iactive+offset)**2
                end do
             end do
          ELSEIF(iactive < ntran) then
             do inich=0, nNiches-1
                start=inich*nChromos/nNiches
                do ipar=start+1,start+offni
                   ichr=rank(ipar)
                   offset=iactive-offtra+(ichr-1)*ntran_IC/nchromos
                   sum=sum+tran_arr(offset)
                   sum2=sum2+tran_arr(offset)**2
                end do
             end do
          ELSE
             do inich=0, nNiches-1
                start=inich*nChromos/nNiches
                do ipar=start+1,start+offni
                   ichr=rank(ipar)
                   offset=iactive-offrota+(ichr-1)*nrota_IC/nchromos
                   sum=sum+rota_arr(offset)
                   sum2=sum2+rota_arr(offset)**2
                end do
                ichr=rank(start+offni)
                offset=iactive-offrota+(ichr-1)*nrota_IC/nchromos
             end do
          ENDIF
          total_dev=total_dev+(sum2/nparents- &
               (sum/nparents)**2)
       end do
    endif
    if(qconve) then
       sum=0.
       sum2=0.
       do inich=0, nNiches-1
          start=inich*nChromos/nNiches
          do ipar=start+1,start+offni
             ichr=rank(ipar)
             sum=sum+energy(ichr)
             sum2=sum2+energy(ichr)**2
          end do
       end do
       total_dev=total_dev+(sum2/nparents- &
            (sum/nparents)**2)
    endif
    return
  end Subroutine Well_Converg
#else /* (genetic_main)*/

  Subroutine genetic_alg(setup, evolve)
    !

    !  procedures.  This routine was written by Charles L. Brooks, III and
    !  Heidi Daigler during the summer of 1994.
    implicit none
    !
    LOGICAL setup, evolve
    !
    CALL WRNDIE(-1,'<GA>','Genetic Algorithm code not compiled.')
    return
  end Subroutine genetic_alg


#endif /* (genetic_main)*/

end module galgor

