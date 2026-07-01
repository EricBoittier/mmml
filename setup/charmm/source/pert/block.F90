module block_fcm
  use chm_kinds
  use block_ltm
   
  implicit none

  integer block_dummy_var

#if KEY_BLOCK==1 /*block_fcm*/

  ! module for block interaction scheme used in free energy calculations
  !-----------------------------------------------------------------------
  !
  ! This module uses variables dealing with block command.
  !
  ! March 2009: Overhauled by Jennifer L. Knight and Charles L Brooks III
  !    * combines variables from ltm/block_ltm.src and 
  !         subroutines from pert/block.src 
  !    * dynamic memory allocation used
  !    * gbblck variables moved from energy/genborn.src 
  !
  !-----------------------------------------------------------------------
  !     QBVSPLT   logical variable stating wheter the VDW block will be
  !               split into repulsive and attractive parts AND scaled
  !               differently.
  !     QDOCK     logical variable stating whether DOCKING is in use
  !                    the SYMMETRIC coefficent for each interaction
  !     BLDOCP    integer pointer into the heap for the array giving
  !                    the ASYMMETRIC coefficent for each interaction
  !     BLENEE, BLENEV
  !               integer pointers into the heap for the arrays giving
  !               block intraction energies
  !     BLNUM     number of accumulated entries in free energy summation
  !     BLSUM     summation for free energy evaluation
  !
  ! for slow growth:
  !     LAMI      initial lambda
  !     LAMF      final lambda
  !     DELLAM    delta lambda per timestep
  !     LSTART    timestep on which to start mutational procedure
  !     LSTOP     timestep on which to end mutational procedure
  !
  !.ab.For HybH. (see AB.J.Comp.Chem2004,25,985-993).
  !     HYBHLB    Hybrid_Hamiltonian Lambda
  !     OUTH      integer stating unit number for dH/dlambda output.
  !.ab.
  ! for fixing and displacing atoms:
  !
  !     QBFIX     logical flag for whether fixed atoms are in operation
  !     BLNFIX    number of fixed atoms in operation
  !     BPXFIX    pointer to fixed x coordinates on heap
  !     BPYFIX    pointer to fixed y coordinates on heap
  !     BPZFIX    pointer to fixed z coordinates on heap
  !     BPIFIX    pointer to array which indexes fixed atoms on heap
  !     QBDIS     logical flag for whether fixed atoms are in operation
  !     BLNDIS    number of fixed atoms in operation
  !     BPXDIS    pointer to displacement x coordinates on heap
  !     BPYDIS    pointer to displacement y coordinates on heap
  !     BPZDIS    pointer to displacement z coordinates on heap
  !     BPIDIS    pointer to array which indexes displacement coordinates on heap
  !     BPJDIS    pointer to array which indexes displacement atoms on heap
  !     QNOBO     logical flag for whether bond term is scaled or not
  !     QNOUB     logical flag for whether urey bradley term is scaled or not
  !     QNOAN     logical flag for whether angle term is scaled or not
  !     QNOIM     logical flag for whether improper torsion term is scaled or not
  !     QNOPH     logical flag for whether proper torsion term is scaled or not
  !     QNOCT     logical flag for whether cmap terms are scaled or not
  !
  LOGICAL, save :: QNOBO, QNOUB, QNOAN, QNOIM, QNOPH, QNOCT
  LOGICAL, save :: QBFIX, QBDIS, QBVSPLT
  INTEGER, save :: LSTART, LSTOP
  INTEGER, save :: BLNFIX
  INTEGER, save :: BLNDIS
  real(chm_real), allocatable, dimension(:), save :: BPXFIX,BPYFIX,BPZFIX
  real(chm_real), allocatable, dimension(:), save :: BPXDIS,BPYDIS,BPZDIS
  integer, allocatable, dimension(:), save :: BPIFIX, BPIDIS, BPJDIS 
  real(chm_real), allocatable, dimension(:), save :: BLENEE, BLENEV
  real(chm_real), save :: BLNUM, BLSUM, LAMI, LAMF, DELLAM
#if KEY_DOCK==1 /*dock_declarations*/
  LOGICAL, save :: QDOCK
  real(chm_real),allocatable,dimension(:), save :: BLDOCP
#endif /* (dock_declarations)*/

  INTEGER, save :: OUTH
!  real(chm_real), save :: HYBHLB ! moved to block_ltm -- Y Huang 2017
  ! Data arrays, logicals and counts for block_exclusions
  logical, save :: qblock_excld, qblock_excld_upinb
  integer,  save :: nblock_excldPairs
  integer, allocatable, dimension(:), save :: block_excldPairs_I, block_excldPairs_J

  !gb_block,gb_lamb,gbldm moved from source/energy/genborn.src
  !  GB_BLOCK    =   Heap pointer to NBLOCK length array containing the GB energy
  !                  which are related to the atoms belong to block n
  !
  !  GB_LAMB     =   Heap pointer to NATOM*NBLOCK lenght array containing the
  !                  atom-based GB energy when IGenType = 1.
  !                  Caustion : GB_LAMB is used only when IgenType = 1.
  !
  !  GBLDM       =   Heap pointer to NBLOCK length array containing the partial
  !                  terms which are used to get the first derivative of
  !                  lambda in lambda-dynamics.
  !                  Caustion : GBLDM is used only when IGenType = 1 and
  !                             QLDM is true.
  real(chm_real),allocatable,dimension(:),save :: &        
       gb_block, gb_lamb, gbldm                           

  !==================================================================
#endif /* (block_fcm)*/

contains
  !==================================================================

  subroutine block_init()
    !=======================================================================
    ! BLOCK.FCM
#if KEY_BLOCK==1 /*block_init*/
    qblock=.false.
    noforc=.false.
    nbsm1=0
    !.ab.THYBH.not used yet (but should be at some point..)
    qhybh=.false.
    !      THYBH=ROOMT
    !.ab.
#if KEY_DOCK==1 /*dock_init*/
    qdock = .false.
#endif /* (dock_init)*/
    qnobo = .false.
    qnoub = .false.
    qnoan = .false.
    qnoim = .false.
    qnoph = .false.
    qnoct = .false.

    nblckcalls = 0
    qsoft_omm = .false.

#endif /* (block_init)*/
    return
  end subroutine block_init

#if KEY_BLOCK==0 /*block_subs*/



  SUBROUTINE BLOCK(COMLYN,COMLEN)
    character(len=*), intent(inout) :: COMLYN
    integer, intent(inout) :: COMLEN
    call wrndie(-1,"Block not compiled","recompile with BLOCK")
    return
  end SUBROUTINE BLOCK

#else /* (block_subs)*/
  SUBROUTINE BLOCK(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     This routine interprets commands dealing with block
    !     evaluation of interactions and free energy calculations.
    !
    !     Bruce Tidor, October 1986
    !     Youngdo Won, 12/18/90
    !
    !     June 2007: Problem with Block and CMAP.
    !                No BLOCK&CMAP for LDM, energy/genborn.src,
    !                ##DOCK yet
    !                Fixed for basic functionality by Milan Hodoscek
    !                Please fix the rest!!! (MF?)
    !
    !     March 2009: * BANBA pref.dat keyword eliminated 
    !                      -all content covered by BLOCK
    !                      -subroutine NOBOAN eliminated
    !                 * LMC, LDLAN, LDMGEN, LRST pref.dat keywords eliminated
    !                      -all content covered by LDM
    !                 * l-dynamics keywords call subroutines in lambdadyn.src
    !                 * l-dynamics variables consolidated into lambdadyn.src
    !                      -includes former blockappend.fcm (for 
    !                       simulated scaling method) and rwlamb.fcm
    !                 * l-dynamics variables updated to use dynamic memory
    !                   allocation 
    !-----------------------------------------------------------------------

    use chm_kinds
    use stream
    use string
    use lambdam    !ldm  !includes rwlamb and blockappend
    use gb_common,only:igentype    
    use fast, only:faster         
#if KEY_SCCDFTB==1
    use blockscc_fcm,only: qsccb            
#endif

    implicit none
    character(len=*), intent(inout) :: COMLYN
    integer, intent(inout) :: COMLEN
    character(len=4) :: wrd
    logical :: qend, qcheck, eof, lused
    INTEGER :: NINT, JINT, I, J, ITEMP, JTEMP
    real(chm_real) :: R, RR
    integer :: alloc_err
    integer :: ltemp

    ! initialize variables 
    EOF=.FALSE.
    QCHECK=.FALSE.
    QEND=.FALSE.
#if KEY_SCCDFTB==1
    qsccb=.false.                                  
#endif

    IF (.NOT. QBLOCK) call blockin(comlyn, comlen)

#if KEY_OPENMM==1
    nblckcalls = nblckcalls + 1
#endif
    !=======================================================================
    !       SUBCOMMANDS FOR BLOCK
    !=======================================================================
    do                         !(continue to read commands until END)
       if (qend) exit
       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
            EOF,.TRUE.,.TRUE.,'BLOCK> ')
       CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
       WRD=NEXTA4(COMLYN,COMLEN)
       IF (WRD  ==  '    ') LUSED=.TRUE.
       if (.not.lused) then   !(big_subcommand_list)

          IF (WRD  ==  'INIT') THEN
             call blockin(comlyn,comlen)
             !
#if KEY_SCCDFTB==1 /*sccdftc*/
          else IF (WRD  ==  'SCCD') THEN
             call block_sccd(comlyn,comlen)
#endif /* (sccdftc)*/
             !
          ELSE IF (WRD  ==  'FREE') THEN
             CALL BLFREE(COMLYN,COMLEN)
             !
             !c New PBLOCK
          ELSE IF (WRD  ==  'PSSP') THEN
! soft core  requires FAST OFF
             if(FASTER > 0) then
               FASTER=-1
               call wrndie(1,'<BLOCK>','BLOCK/PSSP requires FAST OFF; this has been set')  
             endif
             call block_pssp(comlyn,comlen)
             !
          ELSE IF (WRD  ==  'NOPS') THEN
             call block_nopssp
             !
          ELSE IF (WRD  ==  'EAVG') THEN
             CALL BLEAVG(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'COMP') THEN
             CALL BLCOMP(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'SUBC') THEN
             CALL BLSCOMP(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'TRAN') THEN
             CALL BLTRAN(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'ENTR') THEN
             CALL BLEANDS(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'AVER') THEN
             CALL BLAVER(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'RDF ') THEN
             CALL BLRDFC(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'NOFO') THEN
             NOFORC=.TRUE.
             !
          ELSE IF (WRD  ==  'FORC') THEN
             NOFORC=.FALSE.
             !
          ELSE IF (WRD  ==  'NBSM') THEN
             NBSM1=NEXTI(COMLYN,COMLEN)
             !
          ELSE IF (WRD  ==  'COEF') THEN
             !     Procedure SET-COEFFICIENT-SPECIFICALLY
             call block_coef(comlyn,comlen)
             !
          ELSE IF (WRD  ==  'EXCL') THEN
             !      Set-up exclusions between atoms in specified pairs of blocks
             call set_block_exclusions(comlyn,comlen,.false.)
             !
          ELSE IF (WRD  ==  'ADEX') THEN
             !      Set-up exclusions between atoms in specified pairs of blocks
             call set_block_exclusions(comlyn,comlen,.true.)
             !
#if KEY_DOCK==1
          ELSE IF (WRD  ==  'DOCF') THEN
             call block_docf(comlyn,comlen)
#endif /* DOCK*/
#if KEY_OPENMM==1
          ELSE IF (WRD  ==  'SOMM') THEN  ! Set-up usage of soft vdW for OMM using
                                          ! soft core form from Hayes et al., JPC B, 2017
             qsoft_omm = .true.
#endif /* OPENMM*/
             !
          ELSE IF (WRD  ==  'UNSA') THEN
             call block_unsa
             !
          ELSE IF (WRD  ==  'SAVE') THEN
             !       turns on option to save energy components by block
             call block_save
             !
          ELSE IF (WRD  ==  'RMLA') THEN
             !       removes lambda dependence from bond, theta, impr or phi
!(ldm)
             CALL RMLAMBDA
             !
             !------------------------------------------------------------------------
             !  SUBCOMMANDS FOR LAMBDA-DYNAMICS
             !------------------------------------------------------------------------
             ! LDM: L-dynamics using Molecular Dynamics
          ELSE IF (WRD  ==  'QLDM') THEN
             !        turns on LDM
             call ldm_setup(nblock,comlyn,comlen,igentype,gb_lamb,gbldm)
             !
          ELSE IF (WRD  ==  'LDMA') THEN
             !        sets up coefficient matrix   !!!!!!!!!subroutine in BLOCK
             call ldmatrix(qldm .or. qlmc, NBLOCK, ninter, BIXLAM, BLCOEP, &
                  BLCOEB, BLCOEA, BLCOED, BLCOEE, BLCOEV, lstrt)
             !
          ELSE IF (WRD  ==  'LANG') THEN
             !       turns on langevin dynamics to govern dynamics of lambda values 
             call ldm_lang(nblock,comlyn,comlen)

          ELSE IF (WRD  ==  'LDRS') THEN
             !        turns on the reading restart files
             IF(.NOT. QLMC) QLDM=.TRUE. 

          ELSE IF (WRD  ==  'LDWR') THEN
             !        sets up parameters for printing lambda values
             call ldm_write(comlyn,comlen)

          ELSE IF (WRD  ==  'LDIN') THEN
             !        reads in mass, bias, fbeta and initial values for lambdas 
             call ldm_ldin_setup(nblock,comlyn,comlen)

          ELSE IF (WRD  ==  'SOFT') THEN
             !        turn soft cores on or off
             call ldm_scmsld_setup(comlyn,comlen)

          ELSE IF (WRD  ==  'PMEL') THEN
             !        turn MSLD PME on or off
             call ldm_pmemsld_setup(comlyn,comlen)

          ELSE IF (WRD  ==  'SCAT') THEN
             !        turn scaling of constrained atoms on or off
             call ldm_scatmsld_setup(comlyn,comlen)

          ELSE IF (WRD  ==  'CATS') THEN
             !        select atoms that are constrained and may be scaled
             call ldm_call_catsmsld_setup(comlyn,comlen)

          ELSE IF (WRD  ==  'NSOB') THEN
             !        set number of soft bonds
             call ldm_nsoftbond(comlyn,comlen)

          ELSE IF (WRD  ==  'SOBO') THEN
             !        define a particular soft bond
             call ldm_softbond(comlyn,comlen)

          ELSE IF (WRD  ==  'RLFR') THEN
             !        flag for whether to read lambda from restart file
             call ldm_readlambdafromrestart(comlyn,comlen)

          ELSE IF (WRD  ==  'FLAM') THEN
             !        save lambda forces to a variable
             call ldm_save_biflam(comlyn,comlen)

          ELSE IF (WRD  ==  'LDBI') THEN
             !        reads in number of biasing potentials to use
             call ldm_biasin(comlyn,comlen)

          ELSE IF (WRD  ==  'LDBV') THEN
             !        reads in functional form of biasing potentials
             call ldm_biaspot(comlyn,comlen)

          ELSE IF (WRD  ==  'THBV') then
             !        applies a bias to theta
             call ldm_thetabiaspot(comlyn,comlen)

          ELSE IF (WRD  ==  'RSTP') THEN
             !        reads in functional form of restraining potentials
             call ldm_rstp(nblock,comlyn,comlen)

             ! LMC: L-dynamics using hybrid Monte Carlo/Molecular Dynamics
          ELSE IF (WRD  ==  'QLMC') THEN
             !        turns on LMC
             call ldm_mc_setup(nblock,comlyn,comlen)

          ELSE IF (WRD  ==  'MCDI') THEN
             !        specifies step size for lambda movement
             call ldm_mc_increment(comlyn,comlen)

          ELSE IF (WRD  ==  'MCIN') THEN
             !        specifies fixed intermediate values of lambda 
             call ldm_mc_2intermediate(nblock,comlyn,comlen,ltemp)

          ELSE IF (WRD  ==  'MCRS') THEN
             !        turns off lambda force coming from restraining potential
             MCRST = .TRUE.

          ELSE IF (WRD  ==  'MCLE') THEN
             !        turns off hybrid MD/MC
             call ldm_mc_clear

             ! SS: Simulated scaling simulations
          ELSE IF (WRD  ==  'MCFR') THEN
             call ldm_ss_setup(comlyn,comlen)

          ELSE IF (WRD  ==  'MCLA') THEN
             call ldm_ss_lambda(comlyn,comlen)

             !------------------------------------------------------------------------
             !       SUBCOMMANDS FOR MULTI-SITE LAMBDA-DYNAMICS
             !--------------------------------------------------------------------------

          ELSE IF (WRD .EQ. 'MSLD') THEN
             call msld_setup(nblock,comlyn,comlen)

          ELSE IF (WRD .EQ. 'BLAS') THEN    !RLH
             call msld_setup_assign_site(nblock,comlyn,comlen)

          ELSE IF (WRD .EQ. 'MSMA') THEN
             call msld_matrix(nblock, ninter, blcoep)

          ELSE IF (WRD .EQ. 'PHMD') THEN    !GG
             call msld_phmd(comlyn,comlen)  !GG

          ELSE IF (WRD .EQ. 'NORA') THEN    !GG
             call msld_norand               !GG

             !------------------------------------------------------------------------
!ENDIF(ldm)
             !
          ELSE IF (WRD  ==  'CALL') THEN
             call block_call(comlyn,comlen,qcheck)
             !
          ELSE IF (WRD  ==  'LAMB') THEN
             call block_lamb(comlyn,comlen)

             !--------------------------------------------------------------------------
             !       SUBROUTINES FOR USE WITH HYBH
             !--------------------------------------------------------------------------
             !.ab.HybH (see AB.J.Comp.Chem2004,25,985-993).
          ELSE IF (WRD  ==  'HYBH') THEN
             call block_hybh(comlyn,comlen)
             !.ab.
          ELSE IF (WRD  ==  'CLHH') THEN
             if(qhybh)then
                call clearhh()
             endif
             !.ab.
          ELSE IF (WRD  ==  'OUTH') THEN
             !...set output unit for dE/dlambda
             IF (.NOT. QHYBH) CALL WRNDIE(-3,'<BLOCK>', &
                  'OUTH should be used with HybH.')
             OUTH=NEXTI(COMLYN,COMLEN)
             !.ab.For now useless keepit for later use.
          ELSE IF (WRD  ==  'TSTH') THEN
             !...set output unit for dE/dlambda
             IF (.NOT. QHYBH) CALL WRNDIE(-3,'<BLOCK>', &
                  'TSTH should be used with HybH.')
             RR=NEXTF(COMLYN,COMLEN)
             CALL TESTHYBH(RR)
             !.ab.For now useless keepit for later use.
             !      ELSE IF (WRD  ==  'THYB') THEN
             !...set temperature for hybrid mixing.
             !         IF (.NOT. QHYBH) CALL WRNDIE(-3,'<BLOCK>',
             !     $        'THYB should be used with HybH.')
             !         THYBH=NEXTF(COMLYN,COMLEN)
             !.ab.
          ELSE IF (WRD  ==  'PRIN') THEN
             !...print what is ETERMD
             IF (.NOT. QHYBH) CALL WRNDIE(-3,'<BLOCK>', &
                  'PRIN should be used with HybH.')
             CALL HYBHPR(OUTU)
             !.ab.
          ELSE IF (WRD  ==  'PRDH') THEN
             !...print what is ETERMD
             IF (.NOT. QHYBH) CALL WRNDIE(-3,'<BLOCK>', &
                  'PRDH should be used with HybH.')
             CALL PRHYBH()
             !.ab.
          ELSE IF (WRD  ==  'PFOR') THEN
             !...print the forces.
             IF (.NOT. QHYBH) CALL WRNDIE(-3,'<BLOCK>', &
                  'PFOR should be used with HYBrid_Hamiltonian.')
             CALL PRFORC(OUTU)
             !.ab.
             !
          ELSE IF (WRD  ==  'END ') THEN
             call block_end(comlyn,comlen,ltemp,qcheck,qend)
             !
          ELSE IF (WRD  ==  'CLEA') THEN
             call block_clear
             call block_clear_exclusions
             !
          ELSE IF (WRD  ==  'BFIX') THEN
             call block_bfix(comlyn,comlen)
             !
          ELSE IF (WRD  ==  'BDIS') THEN
             call block_bdis(comlyn,comlen)
             !
          ELSE
             CALL WRNDIE(-3,'<BLOCK>','ILLEGAL COMMAND')
          ENDIF
          !
          CALL XTRANE(COMLYN,COMLEN,'BLOCK')
       endif !(big_subcommand_list)

    end do   !(continue reading until END)
    !=======================================================================
    !       END SUBCOMMANDS FOR BLOCK
    !=======================================================================

  END subroutine block

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLFREE(COMLYN,COMLEN)
    !     CALCULATE-FREE-ENERGY-CHANGE
    !
    !     29-Sep-91, Youngdo Won CONT usage overhauled.
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only:qcg           
#endif
  use chm_kinds
  use chm_types
  use memory
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use param_store, only: set_param

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    real(chm_real), external :: EXPC
    INTEGER :: NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    real(chm_real) :: TEMP,BETA,EPREV,EXPO
    real(chm_real) :: OLDL,NEWL,DELL,DELTA,DELTAA
    real(chm_real) :: TEMP1,TEMP2,TEMPY
    INTEGER :: BCNT
    real(chm_real) :: BAVG,BSIG,ACON,SCON
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'

    call chmalloc('block_ltm.src','BLFREE','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLFREE','IFREAT',NATOM,intg=IFREAT)
    OLDL  =GTRMF(COMLYN, COMLEN, 'OLDL',  FMARK)
    NEWL  =GTRMF(COMLYN, COMLEN, 'NEWL',  FMARK)
    NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP) 
    TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    BETA=ONE/(KBOLTZ*TEMP)
    DELL=NEWL-OLDL
    IF (DELL  /=  ZERO .AND. NBLOCK  ==  3) THEN
       BLCOEP(1) = ZERO
       BLCOEB(1) = ZERO
       BLCOEA(1) = ZERO
       BLCOED(1) = ZERO
       BLCOEC(1) = ZERO
       BLCOEV(1) = ZERO
       BLCOEE(1) = ZERO
       BLCOEP(2) = -DELL
       BLCOEB(2) = -DELL
       BLCOEA(2) = -DELL
       BLCOED(2) = -DELL
       BLCOEC(2) = -DELL
       BLCOEV(2) = -DELL
       BLCOEE(2) = -DELL
       BLCOEP(3) = -DELL
       BLCOEB(3) = -DELL
       BLCOEA(3) = -DELL
       BLCOED(3) = -DELL
       BLCOEC(3) = -DELL
       BLCOEV(3) = -DELL
       BLCOEE(3) = -DELL
       BLCOEP(4) = DELL
       BLCOEB(4) = DELL
       BLCOEA(4) = DELL
       BLCOED(4) = DELL
       BLCOEC(4) = DELL
       BLCOEV(4) = DELL
       BLCOEE(4) = DELL
       BLCOEP(5) = ZERO
       BLCOEB(5) = ZERO
       BLCOEC(5) = ZERO
       BLCOEA(5) = ZERO
       BLCOED(5) = ZERO
       BLCOEV(5) = ZERO
       BLCOEE(5) = ZERO
       BLCOEP(6) = DELL
       BLCOEB(6) = DELL
       BLCOEA(6) = DELL
       BLCOED(6) = DELL
       BLCOEC(6) = DELL
       BLCOEV(6) = DELL
       BLCOEE(6) = DELL
    ENDIF
    IF(PRNLEV >= 2) CALL PINTMAT(OUTU)
    !
    !     initialize counter and cummulative variables
    BLSUM=ZERO
    BLNUM=ZERO
    TEMP1=ZERO
    BAVG=ZERO
    BSIG=ZERO
    BCNT=0
    !
    ! 50   CONTINUE
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !
       !     PROCEDURE HANDLE-COORDINATE-ALTERATIONS
       !     FIXED ATOMS
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX, &
            BPZFIX,BPIFIX,BLNFIX)
       !     DISPLACED ATOMS
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS, &
            BPZDIS,BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       !
       IF (NBSM1  ==  0) THEN
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       ELSE
          CALL LOADL(OLDL)
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
          EPREV=EPROP(EPOT)
          CALL LOADL(NEWL)
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
          EPROP(EPOT)=EPROP(EPOT)-EPREV
       ENDIF
       !
       EXPO=-BETA*EPROP(EPOT)
       ACON=EXPC(EXPO)
       SCON=EXPC(EXPO+EXPO)
       !
       !     for cummulation
       BLNUM=BLNUM+ONE
       BLSUM=BLSUM+ACON
       TEMP1=TEMP1+SCON
       !
       IF (NCONT /= 0) THEN
          !     for this BIN
          BCNT=BCNT+1
          BAVG=BAVG+ACON
          BSIG=BSIG+SCON
          !     put the header for statistics output
          IF (INT(BLNUM) == 1 .AND. PRNLEV >= 2) WRITE (OUTU,65)
          IF (MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
             IF (BCNT > 1) THEN
                TEMPY=BAVG/BCNT
                TEMP2=SQRT(BSIG/BCNT-TEMPY*TEMPY)/(BETA*TEMPY)
                TEMPY=-LOG(TEMPY)/BETA
                IF(PRNLEV >= 2) WRITE (OUTU,'(I10,I5,I10,2F15.5)') &
                     INT(BLNUM),BCNT,ISTEP,TEMPY,TEMP2
             ELSE
                TEMPY=-LOG(BAVG)/BETA
                IF(PRNLEV >= 2) WRITE (OUTU,'(I10,I5,I10,F15.5)') &
                     INT(BLNUM),BCNT,ISTEP,TEMPY
             ENDIF
             IF (NCONT < 0) THEN
                BCNT=0
                BAVG=ZERO
                BSIG=ZERO
             ENDIF
          ENDIF
       ENDIF
65     FORMAT(/,5X,'Frame',2X,'Bin',6X,'Step',2X,'Delta Delta A', &
            6X,'Std. Dev.')
       if (istats < 0) exit
    end do

    call chmdealloc('block_ltm.src','BLFREE','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLFREE','IFREAT',NATOM,intg=IFREAT)
    !
    IF (BLNUM <= ZERO) THEN
       CALL WRNDIE(-3,'<BLOCK>','BLNUM is ZERO at the end of BLFREE')
       RETURN
    ENDIF
    !
    !     Cuumulated free energy statistics
    BLSUM=BLSUM/BLNUM
    TEMP1=SQRT(TEMP1/BLNUM-BLSUM*BLSUM)/(BETA*BLSUM)
    DELTAA=-LOG(BLSUM)/BETA
    call set_param('DELTAA',deltaa)
    call set_param('STDDEV',temp1)
    call set_param('BLSUM',blsum)
    IF(PRNLEV >= 2) WRITE(OUTU,75) DELTAA,TEMP,TEMP1,BLSUM
75  FORMAT(' BLOCK> The free energy change is',F12.5,' kcal/mole at', &
         F8.2,' K.',/, &
         '        The standard deviation is',F12.5,/, &
         '        The Normalized SUM is    ',E12.5)

  END subroutine blfree

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLEAVG(COMLYN,COMLEN)
    !     CALCULATE-AVERAGE-ENERGY
    !
    !     31-Sep-91 Youngdo Won
    !     CONT usage overhauled.
    !     Energy terms (and their fluctions) are output by PRINTE calls.
    !     See ENERGY.FCM for energy array definition.
    !     Currently, BOND=1, ANGLE=2, UREYB=3, DIHE=4, IMDIHE=5,
    !     VDW=6, and ELEC=7, so we can loop 1 thru 7 for energy terms.
    !SBblock:
    !     Nov 1994, Stefan Boresch
    !     (1) Make BLOCK work with IMAGE(s)
    !     In order for that to really work, loops from 1 to 7
    !     replaced by loops from 1 to LENENT(=50, currently)
    !     This takes care of IMAGEs if present.
    !     (2) CONT with positive numbers would give wrong results
    !     break up the IF(NCONT /= 0) block into
    !          IF (NCONT < 0)
    !            virtually no changes necessary here (content of old IF
    !            Block)
    !          ELSE IF (NCONT > 0)
    !            do it right this time
    !          ENDIF
    !     (3) to be done: check and make work with CRYSTAL
    !SBb
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg             
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    INTEGER :: NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX

    real(chm_real) :: OLDL,NEWL,DELL,DELTA
    real(chm_real), dimension(lenenp) :: BINEP, CUMEP, BINFP, CUMFP
    real(chm_real), dimension(lenent) :: BINET, CUMET, BINFT, CUMFT
    integer :: I,J,BCNT
    logical :: QHDR
    character(len=4) :: HDRC='CORD', HDRD='VELD'

    !     parsing command items
    call chmalloc('block_ltm.src','BLEAVG','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLEAVG','IFREAT',NATOM,intg=IFREAT)
    OLDL  =GTRMF(COMLYN, COMLEN, 'OLDL',  FMARK)
    NEWL  =GTRMF(COMLYN, COMLEN, 'NEWL',  FMARK)
    NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP) 
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    DELL=NEWL-OLDL
    IF (DELL  /=  ZERO .AND. NBLOCK  ==  3) THEN
       BLCOEP(1) = ZERO
       BLCOEB(1) = ZERO
       BLCOEA(1) = ZERO
       BLCOED(1) = ZERO
       BLCOEC(1) = ZERO
       BLCOEV(1) = ZERO
       BLCOEE(1) = ZERO
       BLCOEP(2) = -DELL
       BLCOEB(2) = -DELL
       BLCOEA(2) = -DELL
       BLCOED(2) = -DELL
       BLCOEC(2) = -DELL
       BLCOEV(2) = -DELL
       BLCOEE(2) = -DELL
       BLCOEP(3) = -DELL
       BLCOEB(3) = -DELL
       BLCOEA(3) = -DELL
       BLCOED(3) = -DELL
       BLCOEC(3) = -DELL
       BLCOEV(3) = -DELL
       BLCOEE(3) = -DELL
       BLCOEP(4) = DELL
       BLCOEB(4) = DELL
       BLCOEA(4) = DELL
       BLCOED(4) = DELL
       BLCOEC(4) = DELL
       BLCOEV(4) = DELL
       BLCOEE(4) = DELL
       BLCOEP(5) = ZERO
       BLCOEB(5) = ZERO
       BLCOEA(5) = ZERO
       BLCOED(5) = ZERO
       BLCOEC(5) = ZERO
       BLCOEV(5) = ZERO
       BLCOEE(5) = ZERO
       BLCOEP(6) = DELL
       BLCOEB(6) = DELL
       BLCOEA(6) = DELL
       BLCOED(6) = DELL
       BLCOEC(6) = DELL
       BLCOEV(6) = DELL
       BLCOEE(6) = DELL
    ENDIF
    IF(PRNLEV >= 2) CALL PINTMAT(OUTU)
    !
    !     initialize local variables
    QHDR = .TRUE.
    BLNUM=ZERO
    BCNT =0
    BINEP=ZERO
    BINFP=ZERO
    CUMEP=ZERO
    CUMFP=ZERO
    BINET=ZERO
    BINFT=ZERO
    CUMET=ZERO
    CUMFT=ZERO

    do 
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                    & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)

       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX, &
            BPZFIX,BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS, &
            BPZDIS,BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       !
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       BLNUM=BLNUM+ONE
       CUMEP(EPOT) = CUMEP(EPOT) + EPROP(EPOT)
       CUMFP(EPOT) = CUMFP(EPOT) + EPROP(EPOT)*EPROP(EPOT)

       do I=1,LENENT
          CUMET(I) = CUMET(I) + ETERM(I)
          CUMFT(I) = CUMFT(I) + ETERM(I)*ETERM(I)
       end do
       !
       IF (NCONT  <  0) THEN
          !     for this BIN
          BCNT=BCNT+1
          BINEP(EPOT) = BINEP(EPOT) + EPROP(EPOT)
          BINFP(EPOT) = BINFP(EPOT) + EPROP(EPOT)*EPROP(EPOT)

          do I=1,LENENT
             BINET(I) = BINET(I) + ETERM(I)
             BINFT(I) = BINFT(I) + ETERM(I)*ETERM(I)
          end do
          !
          !     running average and fluctuation
          IF (MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
             J=INT(BLNUM)
             QHDR=MOD(J,10) == 1
             IF (BCNT > 1) THEN
                BINEP(EPOT)=BINEP(EPOT)/BCNT
                BINFP(EPOT)=SQRT(BINFP(EPOT)/BCNT &
                     -BINEP(EPOT)*BINEP(EPOT))
                BINET=BINET/BCNT
                BINFT=SQRT(BINFT/BCNT-BINET*BINET)
                IF(PRNLEV >= 3) THEN
                   CALL PRINTE(OUTU,BINEP,BINET,'BLEN', 'ENR',QHDR, &
                        J,ZERO,ZERO,.TRUE.)
                   CALL PRINTE(OUTU,BINFP,BINFT,'BLFL', 'ENR',.FALSE., &
                        J,ZERO,ZERO,.TRUE.)
                ENDIF
             ELSE
                IF(PRNLEV >= 3) CALL PRINTE(OUTU,BINEP,BINET,'BLEN', &
                     'ENR',QHDR,J,ZERO,ZERO,.TRUE.)
             ENDIF
             !     reinitialize bin
             BCNT=0
             BINEP(EPOT)=ZERO
             BINFP(EPOT)=ZERO
             BINET=ZERO
             BINFT=ZERO
             IF (QHDR) QHDR=.FALSE.
          ENDIF

       ELSE IF (NCONT > 0) THEN

          IF (MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
             J=INT(BLNUM)
             IF (INT(BLNUM) > 1) THEN
                BINEP(EPOT)=CUMEP(EPOT)/BLNUM
                BINFP(EPOT)=SQRT(CUMFP(EPOT)/BLNUM &
                     -BINEP(EPOT)*BINEP(EPOT))
                BINET=CUMET/BLNUM
                BINFT=SQRT(CUMFT/BLNUM-BINET*BINET)
                IF(PRNLEV >= 3) THEN
                   CALL PRINTE(OUTU,BINEP,BINET,'BLEN', 'ENR',QHDR, &
                        J,ZERO,ZERO,.TRUE.)
                   CALL PRINTE(OUTU,BINFP,BINFT,'BLFL', 'ENR',.FALSE., &
                        J,ZERO,ZERO,.TRUE.)
                ENDIF
             ELSE
                IF(PRNLEV >= 3) CALL PRINTE(OUTU,CUMEP,CUMET,'BLEN', &
                     'ENR',QHDR,J,ZERO,ZERO,.TRUE.)
             ENDIF
             !     ensure that header is printed only once (at the beginning)
             IF (QHDR) QHDR=.FALSE.
          ENDIF
       ENDIF
       !
       if (istats < 0) exit
    end do
    call chmdealloc('block_ltm.src','BLEAVG','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLEAVG','IFREAT',NATOM,intg=IFREAT)

    !     Cummulated average and fluctuation
    IF (BLNUM <= 0.0) THEN
       IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
            ' BLEAVG> ERROR: BLNUM is equal to ZERO.'
       RETURN
    ENDIF
    !
    CUMEP(EPOT)=CUMEP(EPOT)/BLNUM
    CUMFP(EPOT)=SQRT(CUMFP(EPOT)/BLNUM-CUMEP(EPOT)*CUMEP(EPOT))
    CUMET=CUMET/BLNUM
    CUMFT=SQRT(CUMFT/BLNUM-CUMET*CUMET)
    IF(PRNLEV > 3) THEN
       CALL PRINTE(OUTU,CUMEP,CUMET,'ABLE','ENR', &
            .TRUE.,0,ZERO,ZERO,.TRUE.)
       CALL PRINTE(OUTU,CUMFP,CUMFT,'ABLF','ENR', &
            .FALSE.,0,ZERO,ZERO,.TRUE.)
    ENDIF
    !
  END subroutine bleavg

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLCOMP(COMLYN,COMLEN)
    !     Procedure DO-COMPONENT-ANALYSIS
    !
    !      CONT  :   RECORD FREE ENERGY AS FUNCTION OF ENSEMBLE
    !
    !     29-Sep-91 Youngdo Won
    !     Free energy component output via PRINTE
    !     if DELL=0.04, then free energy is printed out in kcal/mol
    !     NOTE: Only Bond, Angle, U-B, Dihedral, Improper, Electrostatic
    !           and van der Waals energy terms are supported by BLOCK.
    !           Consult energy.f90 for ENC definition
    !SBblock:
    !     Nov 1994, Stefan Boresch
    !     (1) Make BLOCK work with IMAGE(s)
    !     In order for that to really work, loops from 1 to 7
    !     replaced by loops from 1 to LENENT(=50, currently)
    !     This takes care of IMAGEs if present.
    !     (2) to be done: check and make work with CRYSTAL
    !SBb
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg              
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    INTEGER :: NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE,NCONT
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    INTEGER :: NDEL,I,J,K,BCNT
    real(chm_real) :: TEMP,BETA,EPREV
    real(chm_real) :: NEWL,DELL,DELTA,FDELL
    real(chm_real), dimension(11) :: ENP, SU
    real(chm_real), dimension(lenent,11) :: ENC 
    real(chm_real), dimension(21) :: ST
    real(chm_real) :: FACT, NEW1
    INTEGER :: NUMI
    LOGICAL :: QHDR
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'
    !
    call chmalloc('block_ltm.src','BLCOMP','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLCOMP','IFREAT',NATOM,intg=IFREAT)
    DELL  =GTRMF(COMLYN, COMLEN, 'DELL',   ZERO)
    NDEL  =GTRMI(COMLYN, COMLEN, 'NDEL',      0)
    FDELL = ONE
    IF (ABS(DELL-0.04) < 0.0001) FDELL=DELL
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP) 
    TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
    NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    IF (NDEL  >  5) CALL WRNDIE(-5,'<BLOCK>', &
         'NDEL cannot exceed 5')
    !
    !     Store the current interaction matrix
    IF (NBLOCK  /=  4) CALL WRNDIE(-5,'<BLOCK>', &
         'Number of blocks must be 4')
    ST(1)=BLCOEP(1)
    ST(2)=BLCOEP(2)
    ST(3)=BLCOEP(3)
    ST(4)=BLCOEP(4)
    ST(5)=BLCOEP(5)
    ST(6)=BLCOEP(6)
    ST(7)=BLCOEP(7)
    ST(8)=BLCOEP(8)
    ST(9)=BLCOEP(9)
    ST(10)=BLCOEP(10)
    ST(11)=ZERO
    ST(12)=-ONE
    ST(13)=-ONE
    ST(14)=ONE
    ST(15)=ZERO
    ST(16)=ONE
    ST(17)=ZERO
    ST(18)=-ONE
    ST(19)=ONE
    ST(20)=ZERO
    IF(PRNLEV >= 2) CALL PINTMAT(OUTU)
    !
    enc = zero
    enp = zero
    su = zero
    !
    BETA=ONE/(KBOLTZ*TEMP)
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    BLNUM=ZERO
    BCNT=0
    !
    !     Loop over dynamics trajectory frames
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                   & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !
       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX, &
            BPZFIX,BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS, &
            BPZDIS,BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       !
       !     Total interaction energy
       blcoep(1:10) = ST(11:20)
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       EPREV=EPROP(EPOT)
       !
       !     Component interaction energy
       blcoep(1:10) = ST(1:10)
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       !
       BLNUM=BLNUM+ONE
       !
       NEWL=NDEL*DELL
       !C      OLDL=-NEWL
       !C      NEWL=NEWL+0.000001
       NUMI=NDEL*2+1

       !C    DO 80 NEW1=OLDL,NEWL,DELL ! fixed to conform with F90
       do I=1,NUMI
          NEW1=DELL*(I-NDEL-1)
          FACT=ONE
          IF (NEW1  /=  ZERO) THEN
             IF (-BETA*EPREV*NEW1  >  -1900.0) THEN
                FACT=EXP(-BETA*EPREV*NEW1)
             ELSE
                FACT=ZERO
             ENDIF
          ENDIF
          SU(I) =SU(I)+FACT
          ENP(I)=ENP(I)+EPROP(EPOT)*FACT
          DO J=1,LENENT
             ENC(J,I)=ENC(J,I)+ETERM(J)*FACT
          ENDDO
          !SBblock: this is a mess, but for pseudo-compatibility
          !SBb      the whole block-printout should be re-considered...
          IF (NCONT /= 0.AND.SU(I) > 0.0000000001) THEN
             IF (INT(BLNUM) == 1 .AND.PRNLEV >= 2) WRITE(OUTU,65) ISTEP
             IF (MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
                IF(PRNLEV >= 2) WRITE (OUTU,'(I7,I5,F8.4,10F12.5,E12.4)') &
                     INT(BLNUM),NCONT,NEW1,ENP(I)/SU(I), &
                     ENC(VDW,I)/SU(I),ENC(ELEC,I)/SU(I), &
                     ENC(BOND,I)/SU(I),ENC(ANGLE,I)/SU(I), &
                     ENC(DIHE,I)/SU(I),ENC(IMDIHE,I)/SU(I), &
                     ENC(UREYB,I)/SU(I),ENC(IMVDW,I)/SU(I), &
                     ENC(IMELEC,I)/SU(I),SU(I)
             ENDIF
          ENDIF
       end do

65     FORMAT('START FROM STEP',I8,/,2X,'Frame',2X,'Bin',4X,'NEWL',2X, &
            'FPOT',2X,'FVDW',2X,'FELE',2X,'FBOND etc. START FROM STEP...')
       if (istats  <  0) exit
    end do
    call chmdealloc('block_ltm.src','BLCOMP','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLCOMP','IFREAT',NATOM,intg=IFREAT)

    !     Write output at each delta lambda step
    !     PRINTE is utilized 29-Sep-91 YW
    IF(PRNLEV >= 2) WRITE (OUTU,'(/,A,F7.2,A,/)') &
         'BLOCK> Change in free energy / lambda at',TEMP,' K.'

    !C    DO 100 NEW1=OLDL,NEWL,DELL
    DO I=1,NUMI
       NEW1=DELL*(I-NDEL-1)
       ETERM(1:LENENT) = ZERO
       ENP(I)=ENP(I)/SU(I)*FDELL
       DO K=1,LENENT
          ENC(K,I)=ENC(K,I)  /SU(I)*FDELL
       ENDDO
       EPROP(EPOT)=ENP(I)
       QHDR=I == 1
       IF(PRNLEV >= 3) THEN
          IF (.NOT.QHDR) WRITE(OUTU,115) NEW1
          !yw.. In    CALL PRINTE(OUTU,EPROP,ENC(J,1),'BLOC', 'ENR',
          !     ENC(J,1) must be ENC(1,I), 03-Aug-95
          CALL PRINTE(OUTU,EPROP,ENC(1,I),'BLOC', 'ENR', &
               QHDR,I,ZERO,DELL,.TRUE.)
       ENDIF
    end do
115 FORMAT('BLOCK> at the delta lambda of',F6.2, &
         ' from the production run.')
    !C      DO 150 NEW1=OLDL,NEWL,DELL
    DO I=2,NUMI
       ENP(1)=ENP(1)+ENP(I)
       DO J=1,LENENT
          ENC(J,1)=ENC(J,1)+ENC(J,I)
       end do
    end do
    EPROP(EPOT)=ENP(1)
    IF(PRNLEV >= 3) THEN
       WRITE (OUTU,'(/,A,/)') 'BLOCK> Cummulated free energy change:'
       CALL PRINTE(OUTU,EPROP,ENC,'BLOC', 'ENR', &
            .TRUE.,0,ZERO,DELL,.TRUE.)
    ENDIF

  END subroutine blcomp

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLSCOMP(COMLYN,COMLEN)
    !     Procedure DO-SUBCOMPONENT-ANALYSIS
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                 
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    real(chm_real),allocatable,dimension(:) :: JTEMP
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    INTEGER :: NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    INTEGER :: NDEL,I,J
    real(chm_real) :: TEMP,BETA,EPREV
    real(chm_real) :: DELL,DELTA
    real(chm_real), dimension(11,9) :: EN
    real(chm_real), dimension(11) :: SU
    real(chm_real), dimension(21) :: ST
    real(chm_real) :: FACT, NEW1
    INTEGER :: NUMI
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'
    !
    call chmalloc('block_ltm.src','BLSCOMP','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLSCOMP','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('block_ltm.src','BLSCOMP','JTEMP',21,crl=JTEMP)
    DELL  =GTRMF(COMLYN, COMLEN, 'DELL',   ZERO)
    NDEL  =GTRMI(COMLYN, COMLEN, 'NDEL',      0)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP) 
    TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    IF (NDEL  >  5) THEN
       CALL WRNDIE(-5,'<BLOCK>','NDEL cannot exceed 5')
    ENDIF
    !
    !     Store the current interaction matrix
    IF (NBLOCK  /=  6) THEN
       CALL WRNDIE(-5,'<BLOCK>','Number of blocks must be 6')
    ENDIF
    ST(1)=  ZERO
    ST(2)= -ONE
    ST(3)= -ONE
    ST(4)=  ONE
    ST(5)=  ZERO
    ST(6)=  ONE
    ST(7)=  ZERO
    ST(8)= -ONE
    ST(9)=  ONE
    ST(10)= ZERO
    ST(11)=-ONE
    ST(12)=-ONE
    ST(13)= ZERO
    ST(14)=-ONE
    ST(15)=-ONE
    ST(16)= ONE
    ST(17)= ZERO
    ST(18)= ONE
    ST(19)= ONE
    ST(20)= ZERO
    ST(21)= ONE
    jtemp(1:21) = ST(1:21)
    ST(1)=BLCOEP(1)
    ST(2)=BLCOEP(2)
    ST(3)=BLCOEP(3)
    ST(4)=BLCOEP(4)
    ST(5)=BLCOEP(5)
    ST(6)=BLCOEP(6)
    ST(7)=BLCOEP(7)
    ST(8)=BLCOEP(8)
    ST(9)=BLCOEP(9)
    ST(10)=BLCOEP(10)
    ST(11)=BLCOEP(11)
    ST(12)=BLCOEP(12)
    ST(13)=BLCOEP(13)
    ST(14)=BLCOEP(14)
    ST(15)=BLCOEP(15)
    ST(16)=BLCOEP(16)
    ST(17)=BLCOEP(17)
    ST(18)=BLCOEP(18)
    ST(19)=BLCOEP(19)
    ST(20)=BLCOEP(20)
    ST(21)=BLCOEP(21)
    IF(PRNLEV >= 2) CALL PINTMAT(OUTU)
    en = zero
    su = zero
    BETA=ONE/(KBOLTZ*TEMP)
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                   & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX,BPZFIX, &
            BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS,BPZDIS, &
            BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       blcoep(1:21) = JTEMP(1:21)
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       EPREV=EPROP(EPOT)
       blcoep(1:21) = ST(1:21)
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       !C      NEWL=NDEL*DELL
       !C      OLDL=-NEWL
       !C      NEWL=NEWL+0.000001
       NUMI=NDEL*2+1

       !C    DO 350   NEW1=OLDL,NEWL,DELL
       do I=1,NUMI
          NEW1=DELL*(I-NDEL-1)
          FACT=ONE
          IF (NEW1  /=  ZERO) FACT=EXP(-BETA*EPREV*NEW1)
          EN(I,1)=EN(I,1)+EPROP(EPOT)*FACT
          EN(I,2)=EN(I,2)+ETERM(BOND)*FACT
          EN(I,3)=EN(I,3)+ETERM(ANGLE)*FACT
          EN(I,4)=EN(I,4)+ETERM(UREYB)*FACT
          EN(I,5)=EN(I,5)+ETERM(DIHE)*FACT
          EN(I,6)=EN(I,6)+ETERM(IMDIHE)*FACT
          EN(I,7)=EN(I,7)+ETERM(VDW)*FACT
          EN(I,8)=EN(I,8)+ETERM(ELEC)*FACT
          EN(I,9)=EN(I,9)+ETERM(HBOND)*FACT
          SU(I)=SU(I)+FACT
       end do
       if (istats  <  0) exit
    end do
    call chmdealloc('block_ltm.src','BLSCOMP','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLSCOMP','IFREAT',NATOM,intg=IFREAT)
    call chmdealloc('block_ltm.src','BLSCOMP','JTEMP',21,crl=JTEMP)
    !
    !     Write output
    do I=1,NUMI
       NEW1=DELL*(I-NDEL-1)
       EN(I,1)=EN(I,1)/SU(I)
       EN(I,2)=EN(I,2)/SU(I)
       EN(I,3)=EN(I,3)/SU(I)
       EN(I,4)=EN(I,4)/SU(I)
       EN(I,5)=EN(I,5)/SU(I)
       EN(I,6)=EN(I,6)/SU(I)
       EN(I,7)=EN(I,7)/SU(I)
       EN(I,8)=EN(I,8)/SU(I)
       EN(I,9)=EN(I,9)/SU(I)
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,360)
360       FORMAT('      ')
          WRITE(OUTU,370) TEMP
370       FORMAT(' Change in free energy / lambda at ', F10.3,' K.')
          WRITE(OUTU,380) NEW1
380       FORMAT(' ...at a delta lambda of ', F8.5)
          WRITE(OUTU,390) EN(I,1)
          WRITE(OUTU,400) EN(I,2)
          WRITE(OUTU,410) EN(I,3)
          WRITE(OUTU,415) EN(I,4)
          WRITE(OUTU,420) EN(I,5)
          WRITE(OUTU,430) EN(I,6)
          WRITE(OUTU,440) EN(I,7)
          WRITE(OUTU,450) EN(I,8)
          WRITE(OUTU,460) EN(I,9)
       ENDIF
390    FORMAT('>>> TOTAL INTERACTION ENERGY: ', F15.5)
400    FORMAT('>>> BOND CONTRIBUTION:        ', F15.5)
410    FORMAT('>>> ANGLE CONTRIBUTION:       ', F15.5)
415    FORMAT('>>> UREY-BRADLEY CONTRIBUTION:', F15.5)
420    FORMAT('>>> DIHEDRAL CONTRIBUTION:    ', F15.5)
430    FORMAT('>>> IMPROPER CONTRIBUTION:    ', F15.5)
440    FORMAT('>>> VDW CONTRIBUTION:         ', F15.5)
450    FORMAT('>>> ELEC CONTRIBUTION:        ', F15.5)
460    FORMAT('>>> HBOND CONTRIBUTION:       ', F15.5)
    end do

  END subroutine blscomp

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLTRAN(COMLYN,COMLEN)
    !     Procedure CALCULATE-TRANSITION-PROBABILITY
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                 
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    real(chm_real), external :: EXPC
    INTEGER :: NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    real(chm_real) :: TEMP,BETA,EPREV,EXPO
    real(chm_real) :: OLDL,NEWL,DELL,DELTA
    real(chm_real) :: TEMP1,TEMP2,TEMPY
    real(chm_real) :: BIAS
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'
    !
    call chmalloc('block_ltm.src','BLTRAN','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLTRAN','IFREAT',NATOM,intg=IFREAT)
    BIAS  =GTRMF(COMLYN, COMLEN, 'BIAS',   ZERO)
    OLDL  =GTRMF(COMLYN, COMLEN, 'OLDL',  FMARK)
    NEWL  =GTRMF(COMLYN, COMLEN, 'NEWL',  FMARK)
    NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
    TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    BETA=ONE/(KBOLTZ*TEMP)
    DELL=NEWL-OLDL
    IF (DELL  /= ZERO .AND. NBLOCK  ==  3) THEN
       BLCOEP(1) = ZERO
       BLCOEB(1) = ZERO
       BLCOEA(1) = ZERO
       BLCOED(1) = ZERO
       BLCOEV(1) = ZERO
       BLCOEE(1) = ZERO
       BLCOEP(2) = -DELL
       BLCOEB(2) = -DELL
       BLCOEA(2) = -DELL
       BLCOED(2) = -DELL
       BLCOEV(2) = -DELL
       BLCOEE(2) = -DELL
       BLCOEP(3) = -DELL
       BLCOEB(3) = -DELL
       BLCOEA(3) = -DELL
       BLCOED(3) = -DELL
       BLCOEV(3) = -DELL
       BLCOEE(3) = -DELL
       BLCOEP(4) = DELL
       BLCOEB(4) = DELL
       BLCOEA(4) = DELL
       BLCOED(4) = DELL
       BLCOEV(4) = DELL
       BLCOEE(4) = DELL
       BLCOEP(5) = ZERO
       BLCOEB(5) = ZERO
       BLCOEA(5) = ZERO
       BLCOED(5) = ZERO
       BLCOEV(5) = ZERO
       BLCOEE(5) = ZERO
       BLCOEP(6) = DELL
       BLCOEB(6) = DELL
       BLCOEA(6) = DELL
       BLCOED(6) = DELL
       BLCOEV(6) = DELL
       BLCOEE(6) = DELL
    ENDIF
    IF(PRNLEV >= 2) CALL PINTMAT(OUTU)
    BLSUM=ZERO
    BLNUM=ZERO
    TEMP1=ZERO
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                    & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX,BPZFIX, &
            BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS,BPZDIS, &
            BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       IF (NBSM1  ==  0) THEN
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       ELSE
          CALL LOADL(OLDL)
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
          EPREV=EPROP(EPOT)
          CALL LOADL(NEWL)
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
          EPROP(EPOT)=EPROP(EPOT)-EPREV
       ENDIF
       EXPO=-BETA*(EPROP(EPOT)+BIAS)
       TEMP2=MIN(ONE,EXPC(EXPO))
       BLSUM=BLSUM+TEMP2
       TEMP1=TEMP1+TEMP2*TEMP2
       BLNUM=BLNUM+ONE
       IF (NCONT  /=  0) THEN
          IF (MOD(INT(BLNUM),ABS(NCONT))  ==  0) THEN
             IF (BLNUM  >  ZERO) THEN
                TEMPY=BLSUM/BLNUM
             ELSE
                TEMPY=ZERO
             ENDIF
             IF(PRNLEV >= 3) WRITE(OUTU, 500) BLNUM, TEMPY
500          FORMAT(' ', 2F10.5)
             IF (NCONT  <  0) THEN
                BLSUM=ZERO
                BLNUM=ZERO
             ENDIF
          ENDIF
       ENDIF
       if (istats  <  0) exit
    end do
    call chmdealloc('block_ltm.src','BLTRAN','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLTRAN','IFREAT',NATOM,intg=IFREAT)

    IF (NCONT  >=  0) THEN
       IF (BLNUM  >  ZERO) THEN
          BLSUM=BLSUM/BLNUM
          TEMP1=TEMP1/BLNUM
          TEMP1=SQRT(TEMP1-BLSUM*BLSUM)
       ELSE
          BLSUM=ZERO
          TEMP1=ZERO
       ENDIF
       IF(PRNLEV >= 3) WRITE(OUTU,510) BLSUM, TEMP
510    FORMAT(' <<<<< The transition probability is ', F10.5, &
            ' kcal/mole at ', F10.4, ' degrees Kelvin. >>>>>')
       IF(PRNLEV >= 3) WRITE(OUTU,520) TEMP1, BIAS
520    FORMAT(' RMS Fluctuation: ', F10.3, '  Bias Energy: ', F12.5)
    ENDIF

  END subroutine bltran

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLEANDS(COMLYN,COMLEN)
    !     Procedure CALCULATE-E-AND-S
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                 
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    INTEGER :: NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    real(chm_real) :: TEMP,BETA,EPREV
    real(chm_real) :: OLDL,NEWL,DELTA,DELTAA
    real(chm_real) :: BETAP,BETAM,BETALP,BETALM,DTEMP
    real(chm_real) :: SUM0,SUM1,SUM2,SUM3,SUM4
    !C      real(chm_real) SUM5,SUM6
    real(chm_real) :: TEMP0,TEMP1,TEMP2,TEMP3,TEMP4,TEMP5,TEMP6,TEMP7
    real(chm_real) :: TEMP8,TEMP9,TEMP10,TEMPY
    real(chm_real) :: NEW1,NEW2,NEW3,NEW4,E1,E2,DELTAE,TDELSN
    real(chm_real), external :: EXPC
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'
    !
    call chmalloc('block_ltm.src','BLEANDS','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLEANDS','IFREAT',NATOM,intg=IFREAT)
    OLDL  =GTRMF(COMLYN, COMLEN, 'OLDL',  FMARK)
    NEWL  =GTRMF(COMLYN, COMLEN, 'NEWL',  FMARK)
    NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
    TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    DTEMP =GTRMF(COMLYN, COMLEN, 'DTEM',    ONE)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    BETA=ONE/(KBOLTZ*TEMP)
    BETAP=ONE/(KBOLTZ*(TEMP+DTEMP))
    BETAM=ONE/(KBOLTZ*(TEMP-DTEMP))
    BETALP=(ONE/(TEMP+DTEMP)-ONE/TEMP)/KBOLTZ
    BETALM=(ONE/(TEMP-DTEMP)-ONE/TEMP)/KBOLTZ
    BLNUM=ZERO
    SUM0=ZERO
    SUM1=ZERO
    SUM2=ZERO
    SUM3=ZERO
    SUM4=ZERO
    !C      SUM5=ZERO
    !C      SUM6=ZERO
    TEMP0=ZERO
    TEMP1=ZERO
    TEMP2=ZERO
    TEMP3=ZERO
    TEMP4=ZERO
    TEMP5=ZERO
    TEMP6=ZERO
    TEMP7=ZERO
    TEMP8=ZERO
    TEMP9=ZERO
    TEMP10=ZERO
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    IF (NBLOCK  /=  3) CALL WRNDIE(-3,'<BLOCK>', &
         'Must use three blocks for this option.')
    !
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                       & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX,BPZFIX, &
            BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS,BPZDIS, &
            BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       CALL LOADL(OLDL)
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       EPREV=EPROP(EPOT)
       CALL LOADL(NEWL)
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       BLNUM=BLNUM+ONE
       TEMPY=EXPC(BETA*(EPREV-EPROP(EPOT)))
       SUM0=SUM0+TEMPY
       TEMP0=TEMP0+TEMPY*TEMPY
       NEW1=EXPC(-EPROP(EPOT)*BETAP+EPREV*BETA)
       SUM1=SUM1+NEW1
       TEMP1=TEMP1+NEW1*NEW1
       NEW2=EXPC(-EPROP(EPOT)*BETAM+EPREV*BETA)
       SUM2=SUM2+NEW2
       TEMP2=TEMP2+NEW2*NEW2
       NEW3=EXPC(-EPREV*BETALP)
       SUM3=SUM3+NEW3
       TEMP3=TEMP3+NEW3*NEW3
       NEW4=EXPC(-EPREV*BETALM)
       SUM4=SUM4+NEW4
       TEMP4=TEMP4+NEW4*NEW4
       TEMP5=TEMP5+NEW1*NEW2
       TEMP6=TEMP6+NEW1*NEW3
       TEMP7=TEMP7+NEW2*NEW4
       TEMP8=TEMP8+NEW3*NEW4
       TEMP9=TEMP9+NEW1*NEW4
       TEMP10=TEMP10+NEW2*NEW3
       if (istats  <  0) exit
    end do
    call chmdealloc('block_ltm.src','BLEANDS','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLEANDS','IFREAT',NATOM,intg=IFREAT)

    IF (NCONT  >=  0) THEN
       SUM0=SUM0/BLNUM
       SUM1=SUM1/BLNUM
       SUM2=SUM2/BLNUM
       SUM3=SUM3/BLNUM
       SUM4=SUM4/BLNUM
       TEMP0=SQRT(TEMP0/BLNUM-SUM0*SUM0)/SUM0
       TEMP1=SQRT(TEMP1/BLNUM-SUM1*SUM1)/SUM1
       TEMP2=SQRT(TEMP2/BLNUM-SUM2*SUM2)/SUM2
       TEMP3=SQRT(TEMP3/BLNUM-SUM3*SUM3)/SUM3
       TEMP4=SQRT(TEMP4/BLNUM-SUM4*SUM4)/SUM4
       TEMP5=(TEMP5/BLNUM-SUM1*SUM2)/(SUM1*SUM2)
       TEMP6=(TEMP6/BLNUM-SUM1*SUM3)/(SUM1*SUM3)
       TEMP7=(TEMP7/BLNUM-SUM2*SUM4)/(SUM2*SUM4)
       TEMP8=(TEMP8/BLNUM-SUM3*SUM4)/(SUM3*SUM4)
       TEMP9=(TEMP9/BLNUM-SUM1*SUM4)/(SUM1*SUM4)
       TEMP10=(TEMP10/BLNUM-SUM2*SUM3)/(SUM2*SUM3)
       DELTAA=-LOG(SUM0)/BETA
       DELTAE=(0.5D0*TEMP/(BETA*DTEMP))*LOG((SUM1/SUM2)*(SUM4/SUM3))
       TDELSN=DELTAA-DELTAE
       EPREV=ABS(TEMP0/BETA)
       E1=(0.5D0*TEMP/(BETA*DTEMP))*SQRT(ABS(TEMP1*TEMP1+TEMP2 &
            *TEMP2+ &
            TEMP3*TEMP3+TEMP4*TEMP4-2.0D0*(TEMP5+TEMP6+TEMP7+TEMP8- &
            TEMP9-TEMP10)))
       E2=SQRT(EPREV*EPREV+E1*E1)
       IF(PRNLEV >= 3) WRITE(OUTU,580) DELTAA, DELTAE, TDELSN, TEMP
580    FORMAT(' <<< DeltaA, DeltaE, and -TDeltaS: ', 3F12.5, &
            ' in kcal/mol  at ', F7.2, ' K>>>')
       IF(PRNLEV >= 3) WRITE(OUTU,590) EPREV, E1, E2
590    FORMAT(' With Standard Dev.  Fluctuations: ', 3F12.5)
    ENDIF

  END subroutine bleands

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLAVER(COMLYN,COMLEN)
    !     Procedure PROCESS-AVERAGE-COMMAND
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use coordc
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    real(chm_real), external :: EXPC
    LOGICAL :: LPERT
    INTEGER :: NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    INTEGER :: I,IATOM,JATOM
    real(chm_real) :: TEMP,BETA,EPREV,EXPO
    real(chm_real) :: OLDL,NEWL,DELL,DELTA
    real(chm_real) :: TEMP1,TEMP3,TEMP4,TEMP5
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'
    character(len=4) :: wrd

    call chmalloc('block_ltm.src','BLAVER','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLAVER','IFREAT',NATOM,intg=IFREAT)
    WRD = NEXTA4(COMLYN,COMLEN)
    IF (WRD  ==  'STRU') THEN
       do I=1,NATOM
          XCOMP(I)=ZERO
          YCOMP(I)=ZERO
          ZCOMP(I)=ZERO
       end do
    ELSE IF (WRD  ==  'DIST') THEN
       IATOM =NEXTI(COMLYN, COMLEN)
       JATOM =NEXTI(COMLYN, COMLEN)
    ELSE
       CALL WRNDIE(-3,'<BLOCK>','ILLEGAL COMMAND')
    ENDIF
    LPERT =(INDXA(COMLYN, COMLEN, 'PERT')  >  0)
    OLDL  =GTRMF(COMLYN, COMLEN, 'OLDL',  FMARK)
    NEWL  =GTRMF(COMLYN, COMLEN, 'NEWL',  FMARK)
    NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
    TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    BETA=ONE/(KBOLTZ*TEMP)
    DELL=NEWL-OLDL
    IF (DELL  /= ZERO .AND. NBLOCK  ==  3) THEN
       BLCOEP(1) = ZERO
       BLCOEB(1) = ZERO
       BLCOEA(1) = ZERO
       BLCOED(1) = ZERO
       BLCOEV(1) = ZERO
       BLCOEE(1) = ZERO
       BLCOEP(2) = -DELL
       BLCOEB(2) = -DELL
       BLCOEA(2) = -DELL
       BLCOED(2) = -DELL
       BLCOEV(2) = -DELL
       BLCOEE(2) = -DELL
       BLCOEP(3) = -DELL
       BLCOEB(3) = -DELL
       BLCOEA(3) = -DELL
       BLCOED(3) = -DELL
       BLCOEV(3) = -DELL
       BLCOEE(3) = -DELL
       BLCOEP(4) = DELL
       BLCOEB(4) = DELL
       BLCOEA(4) = DELL
       BLCOED(4) = DELL
       BLCOEV(4) = DELL
       BLCOEE(4) = DELL
       BLCOEP(5) = ZERO
       BLCOEB(5) = ZERO
       BLCOEA(5) = ZERO
       BLCOED(5) = ZERO
       BLCOEV(5) = ZERO
       BLCOEE(5) = ZERO
       BLCOEP(6) = DELL
       BLCOEB(6) = DELL
       BLCOEA(6) = DELL
       BLCOED(6) = DELL
       BLCOEV(6) = DELL
       BLCOEE(6) = DELL
    ENDIF
    IF (LPERT .AND. PRNLEV >= 2) CALL PINTMAT(OUTU)
    BLSUM=ZERO
    BLNUM=ZERO
    TEMP1=ZERO
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                     & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX,BPZFIX, &
            BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS,BPZDIS, &
            BPIDIS,BPJDIS,BLNDIS)
       IF (LPERT) THEN
          CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
          IF (NBSM1  ==  0) THEN
             CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
          ELSE
             CALL LOADL(OLDL)
             CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
             EPREV=EPROP(EPOT)
             CALL LOADL(NEWL)
             CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
             EPROP(EPOT)=EPROP(EPOT)-EPREV
          ENDIF
          EXPO=EXPC(-BETA*EPROP(EPOT))
       ELSE
          EXPO=ONE
       ENDIF
       IF (WRD  ==  'STRU') THEN
          do I=1, NATOM
             XCOMP(I)=XCOMP(I)+X(I)*EXPO
             YCOMP(I)=YCOMP(I)+Y(I)*EXPO
             ZCOMP(I)=ZCOMP(I)+Z(I)*EXPO
          end do
          BLNUM=BLNUM+EXPO
       ELSE IF (WRD  ==  'DIST') THEN
          TEMP3=(X(IATOM)-X(JATOM))*(X(IATOM)-X(JATOM))
          TEMP4=(Y(IATOM)-Y(JATOM))*(Y(IATOM)-Y(JATOM))
          TEMP5=(Z(IATOM)-Z(JATOM))*(Z(IATOM)-Z(JATOM))
          BLSUM=BLSUM+SQRT(TEMP3+TEMP4+TEMP5)*EXPO
          BLNUM=BLNUM+EXPO
       ENDIF
       if (istats  <  0) exit
    end do
    call chmdealloc('block_ltm.src','BLAVER','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLAVER','IFREAT',NATOM,intg=IFREAT)

    IF (WRD  ==  'STRU') THEN
       do I=1, NATOM
          X(I)=XCOMP(I)/BLNUM
          Y(I)=YCOMP(I)/BLNUM
          Z(I)=ZCOMP(I)/BLNUM
          XCOMP(I)=ZERO
          YCOMP(I)=ZERO
          ZCOMP(I)=ZERO
       end do
       IF(PRNLEV >= 3) WRITE(OUTU,650)
650    FORMAT(' Average Structure has been put into main ', &
            /, 'coordinate set.  Comparison coordinate set has ', &
            /, 'been zeroed.')
    ELSE IF (WRD  ==  'DIST') THEN
       BLSUM=BLSUM/BLNUM
       IF(PRNLEV >= 3) WRITE(OUTU,660) BLSUM
660    FORMAT(' <<<<< Average Value is ', F10.5, ' >>>>>')
    ENDIF

  END subroutine blaver

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLRDFC(COMLYN,COMLEN)
    !     Procedure CALCULATE-RADIAL-DISTRIBUTION-FUNCTION
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg             
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use select
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use memory

    implicit none
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    integer,allocatable,dimension(:) :: IFREAT
    integer,allocatable,dimension(:) :: JTEMP
    integer,allocatable,dimension(:) :: BIN
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    INTEGER :: NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
    INTEGER :: ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER :: IHBFRX,INBFRX,ILBFRX,IMGFRX
    INTEGER :: I,J,IATOM,NBIN
    real(chm_real) :: FBIN,SBIN
    real(chm_real) :: XRDF,YRDF,ZRDF,DELTA
    character(len=4) :: HDRC = 'CORD', HDRD = 'VELD'
    !
    call chmalloc('block_ltm.src','BLRDFC','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('block_ltm.src','BLRDFC','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('block_ltm.src','BLRDFC','JTEMP',NATOM,intg=JTEMP)
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
    FBIN  =GTRMF(COMLYN, COMLEN, 'FBIN', ONEPT5)
    NBIN  =GTRMI(COMLYN, COMLEN, 'NBIN',    100)
    SBIN  =GTRMF(COMLYN, COMLEN, 'SBIN',  PT125)
    XRDF  =GTRMF(COMLYN, COMLEN, 'X   ',   ZERO)
    YRDF  =GTRMF(COMLYN, COMLEN, 'Y   ',   ZERO)
    ZRDF  =GTRMF(COMLYN, COMLEN, 'Z   ',   ZERO)
    IATOM =GTRMI(COMLYN, COMLEN, 'ATOM',      0)
    IHBFRX=GTRMI(COMLYN, COMLEN, 'IHBF',      0)
    INBFRX=GTRMI(COMLYN, COMLEN, 'INBF',      0)
    ILBFRX=GTRMI(COMLYN, COMLEN, 'ILBF',      0)
    IMGFRX=GTRMI(COMLYN, COMLEN, 'IMGF',      0)
    call chmalloc('block_ltm.src','BLRDFC','BIN',NBIN,intg=BIN)
    CALL SELCTA(COMLYN,COMLEN,JTEMP,X,Y,Z,WMAIN,.TRUE.)
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    ISTATS=1
    NAT=NATOM
    IUNIT=NFU
    NFREAT=0
    CALL RDFCLC(X,JTEMP,NATOM,FBIN,SBIN,NBIN,BIN,-1)
    J=0
    do
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                                   & 
#endif
            ITEMP,NAT,IFREAT,NFREAT,NFU, &
            NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
            DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       !     Procedure HANDLE-COORDINATE-ALTERATIONS
       !     Fixed Atoms
       IF (QBFIX) CALL BFIX3(BPXFIX,BPYFIX,BPZFIX, &
            BPIFIX,BLNFIX)
       !     Displaced Atoms
       IF (QBDIS) CALL BDIS3(BPXDIS,BPYDIS,BPZDIS, &
            BPIDIS,BPJDIS,BLNDIS)
       CALL BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
       IF (IATOM  >  0) THEN
          XRDF=X(IATOM)
          YRDF=Y(IATOM)
          ZRDF=Z(IATOM)
       ENDIF
       do I=1,NATOM
          X(I)=SQRT((X(I)-XRDF)*(X(I)-XRDF)+(Y(I)-YRDF)* &
               (Y(I)-YRDF)+(Z(I)-ZRDF)*(Z(I)-ZRDF))
       end do
       CALL RDFCLC(X,JTEMP,NATOM,FBIN,SBIN,NBIN,BIN,0)
       J=J+1
       if (istats  <  0) exit
    end do
    CALL RDFCLC(X,JTEMP,NATOM,FBIN,SBIN,NBIN,BIN,J)
    call chmdealloc('block_ltm.src','BLRDFC','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('block_ltm.src','BLRDFC','IFREAT',NATOM,intg=IFREAT)
    call chmdealloc('block_ltm.src','BLRDFC','JTEMP',NATOM,intg=JTEMP)
    call chmdealloc('block_ltm.src','BLRDFC','BIN',NBIN,intg=BIN)

  END subroutine blrdfc

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLINIT(NATOM,NBLOCK,NINT, BLCOP, &
       BLCOB,BLCOA,BLCOD,BLCOC,BLCOE,BLCOV, &
       BLCOVR,BLCOVA,IBLOCK)
    !     THIS ROUTINE INITIALIZES THE SYSTEM TO ALL INTERACTION COEFFICI
    !     EQUAL TO ONE AND ALL ATOMS BELONGING TO BLOCK NUMBER ONE.
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use number

    implicit none
    INTEGER, intent(in) :: NATOM,NBLOCK,NINT
    INTEGER, dimension(natom), intent(out) :: IBLOCK
    real(chm_real), dimension(nint), intent(inout) :: BLCOP,BLCOB,BLCOA, &
         BLCOD, BLCOE, BLCOV, BLCOVR, BLCOVA, BLCOC
    !     LOCAL
    INTEGER :: I
    !
    !     BEGIN
    iblock = 1

    BLCOP=ONE
    BLCOB=ONE
    BLCOA=ONE
    BLCOD=ONE
    BLCOC=ONE
    BLCOE=ONE
    BLCOV=ONE
    BLCOVR=ONE
    BLCOVA=ONE

  END subroutine blinit

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLCHEK(IBLOCK)
    !     THIS ROUTINE CHECKS FOR WHETHER EACH THREE- AND FOUR-BODY TERM
    !     IN THE POTENTIAL FUNCTION (CURRENTLY ANGLES, DIHEDRALS, AND
    !     IMPROPERS)
    !     IS AN INTERACTION BETWEEN AT MOST TWO BLOCKS.
    !
    !     Jay Banks 30 Nov 95:
    !     added ( ##IF MMFF) check of out-of-plane angles.
    !     (Stretch-bend term involves the same atom triplets as bond
    !     angles, so it's already covered.)
    !
    !     June 2007, Milan Hodoscek: Do this also for CMAP
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  use ffieldm

    implicit none
    INTEGER, dimension(natom), intent(in) :: IBLOCK
    INTEGER :: I, ITEMP, JTEMP, KTEMP
#if KEY_CMAP==1
    INTEGER, dimension(8) :: IDIFCT, ITEMPCT    
#endif
#if KEY_CMAP==1
    integer :: IPT, J, K                        
#endif

    !     BEGIN
    !
    !     CHECK THETA
    do I=1, NTHETA
       ITEMP=IBLOCK(IT(I))
       JTEMP=IBLOCK(JT(I))
       KTEMP=IBLOCK(KT(I))
       IF (ITEMP  /=  JTEMP) THEN
          IF (ITEMP  /=  KTEMP .AND. JTEMP  /=  KTEMP)  THEN
             CALL WRNDIE(-3,'<BLCHEK>','ILLEGAL BLOCKING.')
          ENDIF
       ENDIF
#if KEY_MMFF==1 /*mmff*/
       if(ffield == mmff) then
          IF(LTHETA(I) > 0) THEN
             KTEMP=IBLOCK(LTHETA(I))
             IF (ITEMP  ==  JTEMP) THEN
                JTEMP=KTEMP
             ELSE
                IF (ITEMP  /=  KTEMP .AND. JTEMP  /=  KTEMP)  THEN
                   CALL WRNDIE(-3,'<BLCHEK>', &
                        'ILLEGAL BLOCKING FOR MMFF OUT-OF-PLANE ANGLE.')
                ENDIF
             ENDIF
          ENDIF
          endif
#endif /* (mmff)*/
    end do
    !
    !     CHECK PHI
    do I=1, NPHI
       ITEMP=IBLOCK(IP(I))
       JTEMP=IBLOCK(JP(I))
       KTEMP=IBLOCK(KP(I))
       IF (ITEMP  ==  JTEMP) THEN
          JTEMP=KTEMP
       ELSE
          IF (ITEMP  /=  KTEMP .AND. JTEMP  /=  KTEMP)  THEN
             CALL WRNDIE(-3,'<BLCHEK>','ILLEGAL BLOCKING.')
          ENDIF
       ENDIF
       KTEMP=IBLOCK(LP(I))
       IF (ITEMP  ==  JTEMP) THEN
          JTEMP=KTEMP
       ELSE
          IF (ITEMP  /=  KTEMP .AND. JTEMP  /=  KTEMP)  THEN
             CALL WRNDIE(-3,'<BLCHEK>','ILLEGAL BLOCKING.')
          ENDIF
       ENDIF
    end do
    !
    !     CHECK IMPHI
    do I=1, NIMPHI
       ITEMP=IBLOCK(IM(I))
       JTEMP=IBLOCK(JM(I))
       KTEMP=IBLOCK(KM(I))
       IF (ITEMP  ==  JTEMP) THEN
          JTEMP=KTEMP
       ELSE
          IF (ITEMP  /=  KTEMP .AND. JTEMP  /=  KTEMP)  THEN
             CALL WRNDIE(-3,'<BLCHEK>','ILLEGAL BLOCKING.')
          ENDIF
       ENDIF
       KTEMP=IBLOCK(LM(I))
       IF (ITEMP  ==  JTEMP) THEN
          JTEMP=KTEMP
       ELSE
          IF (ITEMP  /=  KTEMP .AND. JTEMP  /=  KTEMP)  THEN
             CALL WRNDIE(-3,'<BLCHEK>','ILLEGAL BLOCKING.')
          ENDIF
       ENDIF
    end do
    !
    !     CHECK CMAP
#if KEY_CMAP==1 /*cmap*/
    do I=1, NCRTERM
       IPT=1
       IDIFCT(IPT)=IBLOCK(I1CT(I))
       ITEMPCT(2)=IBLOCK(J1CT(I))
       ITEMPCT(3)=IBLOCK(K1CT(I))
       ITEMPCT(4)=IBLOCK(L1CT(I))
       ITEMPCT(5)=IBLOCK(I2CT(I))
       ITEMPCT(6)=IBLOCK(J2CT(I))
       ITEMPCT(7)=IBLOCK(K2CT(I))
       ITEMPCT(8)=IBLOCK(L2CT(I))
       outer: DO J=2,8 
          inner: DO K=1,IPT
             IF(ITEMPCT(J) == IDIFCT(K)) cycle outer 
          END DO inner
          IPT=IPT+1
          IDIFCT(IPT)=ITEMPCT(J)
       END DO outer
       IF(IPT > 2)CALL WRNDIE(-3,'<BLCHEK>','ILLEGAL BLOCKING.')
    end do
#endif /* (cmap)*/

  END subroutine blchek

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE RDFCLC(R,JTEMP,NATOM,FBIN,SBIN,NBIN,BIN,ISTAT)
    !     DO RADIAL DISTRIBUTION FUNCTION CALCULATION AND PRINTING.
    !     ISTAT=   <0     INITIALIZE
    !     0     ACCUMULATE
    !     >0     NORMALIZE AND PRINT
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream

    implicit none
    INTEGER, intent(in) :: NBIN, NATOM, ISTAT
    integer, dimension(nbin), intent(out) :: bin
    integer, dimension(natom), intent(in) :: jtemp
    real(chm_real), dimension(natom), intent(in) :: R
    real(chm_real), intent(in) :: SBIN, FBIN
    real(chm_real) :: RAD, VAL
    INTEGER :: I, K

    !     BEGIN
    IF (ISTAT  <  0) THEN
       bin = 0
    ELSE IF (ISTAT  ==  0) THEN
       do I=1, NATOM
          IF (JTEMP(I) == 1) THEN
             K=INT((R(I)-FBIN+SBIN)/SBIN)
             IF (K  >  0 .AND. K  <=  NBIN) BIN(K)=BIN(K)+1
          ENDIF
       end do
    ELSE IF (ISTAT  >  0) THEN
       do I=1, NBIN
          RAD=I*SBIN+FBIN-(SBIN/TWO)
          VAL=BIN(I)/(ISTAT*FOUR*PI*RAD*RAD*SBIN)
          IF(PRNLEV >= 3) WRITE(OUTU,30) RAD, VAL
30        FORMAT(1X,2F10.5)
       end do
    ENDIF

  END subroutine rdfclc

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BFIX1(COMLYN,COMLEN,ITEMP,NATOM)
    !             called by block_bfix
    !     PICK ATOM SELECTION FOR ATOMS TO FIX IN THE EVALUATION OF
    !     FREE ENERGIES.  PASS CONTROL TO BFIX2 IF ATOM SELECTION IS
    !     NOT NULL.
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use coord
    use memory
    use select

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer, intent(in) ::  natom
    INTEGER, dimension(natom), intent(in) :: ITEMP
    integer :: i

    !     BEGIN
    CALL SELCTA(COMLYN,COMLEN,ITEMP,X,Y,Z,WMAIN,.TRUE.)
    BLNFIX=0
    do I=1,NATOM
       IF (ITEMP(I)  ==  1) BLNFIX=BLNFIX+1
    end do
    IF (BLNFIX  >  0) THEN
       call chmalloc('block_ltm.src','BFIX1','BPXFIX',BLNFIX,crl=BPXFIX)
       call chmalloc('block_ltm.src','BFIX1','BPYFIX',BLNFIX,crl=BPYFIX)
       call chmalloc('block_ltm.src','BFIX1','BPZFIX',BLNFIX,crl=BPZFIX)
       call chmalloc('block_ltm.src','BFIX1','BPIFIX',BLNFIX,intg=BPIFIX)
       QBFIX=.TRUE.
       CALL BFIX2(ITEMP,BPXFIX,BPYFIX,BPZFIX,BPIFIX,NATOM)
    ENDIF

  END subroutine bfix1

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BFIX2(ITEMP,BLXFIX,BLYFIX,BLZFIX,BLIFIX,NATOM)
    !     STORES FIXED COORDINATES IN ARRAYS.
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord

    implicit none
    integer, intent(in) :: natom
    integer, intent(in), dimension(natom) :: itemp
    real(chm_real), dimension(natom), intent(out) :: BLXFIX, BLYFIX, BLZFIX
    integer, dimension(natom), intent(out) :: BLIFIX
    integer :: i, j

    !     BEGIN
    J=0
    do I=1,NATOM
       IF (ITEMP(I)  ==  1) THEN
          J=J+1
          BLXFIX(J)=X(I)
          BLYFIX(J)=Y(I)
          BLZFIX(J)=Z(I)
          BLIFIX(J)=I
       ENDIF
    end do

  END subroutine bfix2

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BFIX3(BLXFIX,BLYFIX,BLZFIX,BLIFIX,NFIX)
    !     COPIES FIXED COORDINATES FROM STORAGE ARRAYS (ON HEAP) INTO ACTUAL
    !     MAIN COORDINATE ARRAYS.
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord

    implicit none
    integer, intent(in) :: nfix
    integer, dimension(nfix), intent(in) :: BLIFIX
    real(chm_real), dimension(nfix), intent(in) :: BLXFIX, BLYFIX, BLZFIX
    integer :: I

    !     BEGIN
    do I=1, NFIX
       X(BLIFIX(I))=BLXFIX(I)
       Y(BLIFIX(I))=BLYFIX(I)
       Z(BLIFIX(I))=BLZFIX(I)
    end do

  END subroutine bfix3

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BDIS1(COMLYN,COMLEN,ITEMP,JTEMP,NATOM)
    !     PICK ATOM SELECTIONS FOR ATOMS IN DISPLACEMENT FOR THE EVALUATION
    !     FREE ENERGIES.  PASS CONTROL TO BDIS2 IF ATOM SELECTIONS ARE
    !     NOT NULL.
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use coord
    use memory
    use select

    implicit none
    INTEGER ITEMP(*), JTEMP(*)
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN, NATOM
    integer :: i, j 

    !     BEGIN
    CALL SELCTA(COMLYN,COMLEN,ITEMP,X,Y,Z,WMAIN,.TRUE.)
    CALL SELCTA(COMLYN,COMLEN,JTEMP,X,Y,Z,WMAIN,.TRUE.)
    BLNDIS=0
    J=0
    do I=1,NATOM
       IF (ITEMP(I)  ==  1) BLNDIS=BLNDIS+1
       IF (JTEMP(I)  ==  1) J=J+1
    end do
    IF (J  /=  BLNDIS) THEN
       CALL WRNDIE(-3,'<BDIS1>', &
            'PAIR OF SELECTIONS MUST CONTAIN THE SAME NUMBER OF ATOMS.')
    ENDIF
    IF (BLNDIS  >  0) THEN
       call chmalloc('block_ltm.src','BDIS1','BPXDIS',BLNDIS,crl=BPXDIS)
       call chmalloc('block_ltm.src','BDIS1','BPYDIS',BLNDIS,crl=BPYDIS)
       call chmalloc('block_ltm.src','BDIS1','BPZDIS',BLNDIS,crl=BPZDIS)
       call chmalloc('block_ltm.src','BDIS1','BPIDIS',BLNDIS,intg=BPIDIS)
       call chmalloc('block_ltm.src','BDIS1','BPJDIS',BLNDIS,intg=BPJDIS)
       QBDIS=.TRUE.
       CALL BDIS2(ITEMP,JTEMP,BPXDIS,BPYDIS,BPZDIS, &
            BPIDIS,BPJDIS,NATOM)
    ENDIF

  END subroutine bdis1
  !insert page break

  !-----------------------------------------------------------------------
  SUBROUTINE BDIS2(ITEMP,JTEMP,BLXDIS,BLYDIS,BLZDIS,BLIDIS,BLJDIS, &
       NATOM)
    !     STORES DISPLACEMENT COORDINATES IN ARRAYS.
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord

    implicit none
    integer, intent(in) :: natom
    integer, dimension(natom), intent(in) :: itemp, jtemp
    integer, dimension(natom), intent(out) :: BLIDIS, BLJDIS
    real(chm_real), dimension(natom), intent(out) :: BLXDIS, BLYDIS, BLZDIS
    integer :: i, j, k

    !     BEGIN
    J=0
    K=0
    do I=1,NATOM
       IF (ITEMP(I)  ==  1) THEN
          J=J+1
          BLXDIS(J)=X(I)
          BLYDIS(J)=Y(I)
          BLZDIS(J)=Z(I)
          BLIDIS(J)=I
       ENDIF
       IF (JTEMP(I)  ==  1) THEN
          K=K+1
          BLJDIS(K)=I
       ENDIF
    end do

  END subroutine bdis2

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BDIS3(BLXDIS,BLYDIS,BLZDIS,BLIDIS,BLJDIS,NDIS)
    !     COPIES DISPLACEMENT COORDINATES FROM STORAGE ARRAYS (ON HEAP)
    !     AND ADDS TO MAIN COORDINATE ARRAYS.
    !
    !     INPUT/OUTPUT
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord

    implicit none
    integer, intent(in) :: ndis
    integer, dimension(ndis), intent(in) :: BLIDIS, BLJDIS
    real(chm_real), dimension(ndis), intent(in) :: BLXDIS, BLYDIS, BLZDIS
    integer :: i

    !     BEGIN
    do I=1, NDIS
       X(BLJDIS(I))=X(BLIDIS(I))
       Y(BLJDIS(I))=Y(BLIDIS(I))
       Z(BLJDIS(I))=Z(BLIDIS(I))
    end do

  END subroutine bdis3

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE BLOCKIN(COMLYN,COMLEN)
    !           invoked by BLOCK command "BLOCK" and "INIT"
    !     PROCEDURE DO-INITIALIZATION
    !-----------------------------------------------------------------------
  use chm_kinds
  use memory
  use stream
  use string
  use psf
  use lambdam       !ldm
  use gb_common,only: igentype,alph_gb,sigx_gb,sigy_gb,sigz_gb,t_gb 

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    INTEGER :: NINT, NDOC
    integer :: alloc_err

    IF (QBLOCK) THEN
       NINT=NBLOCK*(NBLOCK+1)/2
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEP',NINT,crl=BLCOEP)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEB',NINT,crl=BLCOEB)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEA',NINT,crl=BLCOEA)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEC',NINT,crl=BLCOEC)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOED',NINT,crl=BLCOED)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEE',NINT,crl=BLCOEE)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEV',NINT,crl=BLCOEV)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEVR',NINT,crl=BLCOEVR)
       call chmdealloc('block_ltm.src','BLOCKIN','BLCOEVA',NINT,crl=BLCOEVA)
       call chmdealloc('block_ltm.src','BLOCKIN','BLENEE',NINT,crl=BLENEE)
       call chmdealloc('block_ltm.src','BLOCKIN','BLENEV',NINT,crl=BLENEV)
#if KEY_DOCK==1
       NDOC = NBLOCK*NBLOCK
       call chmdealloc('block_ltm.src','BLOCKIN','BLDOCP',NDOC,crl=BLDOCP)
#endif /*  DOCK*/
       call ldm_block_clear(nblock,nreplica)  !ldm
    ENDIF

    NBLOCK=0
    NBLOCK=NEXTI(COMLYN,COMLEN)
    NREPLICA=GTRMI(COMLYN,COMLEN,'NREP',1)   !GG Set number of replicas, default is one
    IF (NBLOCK  ==  0) NBLOCK=3
    ninter = NBLOCK*(NBLOCK+1)/2

    IF (NBLOCK  >  0) THEN
       NINT=NBLOCK*(NBLOCK+1)/2
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEP',NINT,crl=BLCOEP)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEB',NINT,crl=BLCOEB)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEA',NINT,crl=BLCOEA)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOED',NINT,crl=BLCOED)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEC',NINT,crl=BLCOEC)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEE',NINT,crl=BLCOEE)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEV',NINT,crl=BLCOEV)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEVR',NINT,crl=BLCOEVR)
       call chmalloc('block_ltm.src','BLOCKIN','BLCOEVA',NINT,crl=BLCOEVA)
       call chmalloc('block_ltm.src','BLOCKIN','BLENEE',NINT,crl=BLENEE)
       call chmalloc('block_ltm.src','BLOCKIN','BLENEV',NINT,crl=BLENEV)
#if KEY_DOCK==1
       NDOC = NBLOCK*NBLOCK
       call chmalloc('block_ltm.src','BLOCKIN','BLDOCP',NDOC,crl=BLDOCP)
#endif /* DOCK*/
       !SBblock: allocate the iblckp/iblock array for maxaim instead of
       !         natom atoms!
       if(.not.qblock) call chmalloc('block_ltm.src','BLOCKIN','IBLCKP',MAXAIM,intg=IBLCKP)
       !SBb

       CALL BLINIT(NATOM,NBLOCK,NINT,BLCOEP,BLCOEB, &
            BLCOEA,BLCOED,BLCOEC,BLCOEE, &
            BLCOEV,BLCOEVR,BLCOEVA,IBLCKP)
#if KEY_DOCK==1
       IF(QDOCK) THEN
          CALL BLINIT(NATOM,NDOC,NINT,BLCOEP,BLCOEB, &
               BLCOEA,BLCOED,BLCOEC,BLCOEE, &
               BLCOEV,BLCOEVR,BLCOEVA,IBLCKP)
       ENDIF
#endif 
       IF(PRNLEV >= 3) THEN
          WRITE(OUTU,55) NBLOCK
          WRITE(OUTU,65)
          WRITE(OUTU,75)
       ENDIF
!ldm
       !        ALLOCATE heap for lambdas
       IF (.NOT.QLDM) call ldm_blockin(nblock,ninter,nreplica) !GG
! LDM
       nblock_excldPairs = 0
       if( prnlev >= 3 ) then
          WRITE(OUTU,*) ' Setting number of block exclusions nblock_excldPairs=0'
       endif
    ENDIF
55  FORMAT(' Block structure initialized with', I4, ' blocks.')
65  FORMAT(' All atoms have been assigned to block 1.')
75  FORMAT(' All interaction coefficients have been set to unity.')

    IF (NBLOCK  >  0) QBLOCK=.TRUE.
    IF (QBLOCK) THEN
       IF (IGenType  ==  2) THEN
          deallocate(alph_gb,sigx_gb,sigy_gb,sigz_gb,t_gb, &
               stat=alloc_err)
          if(alloc_err /= 0) call wrndie(-4,"<block.src>BLOCK", &
               "A: Cannot deallocate  gb arrays")

          allocate(sigx_gb(natom*nblock),sigy_gb(natom*nblock), &
               sigz_gb(natom*nblock),t_gb(natom*nblock), &
               gb_block(nblock), &
               stat=alloc_err)
          if(alloc_err /= 0) call wrndie(-4,"<block.src>BLOCK", &
               "A: Cannot allocate memory for sigx_gb to gb_block")

       ELSE IF (IGenType  ==  1) THEN
          allocate(gb_block(nblock),stat=alloc_err)
          if(alloc_err /= 0) call wrndie(-4,"<block.src>BLOCK", &
               "B: Cannot allocate memory for gb_block")
       ENDIF
    ENDIF
  END subroutine blockin

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE PINTMAT(OUTU)
    !     PROCEDURE PRINT-INTERACTION-MATRIX
    !-----------------------------------------------------------------------
    use chm_kinds

    implicit none
    INTEGER OUTU
    INTEGER I,J,K
    !
    !
    WRITE(OUTU,15) 'Matrix of Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    do I=1,NBLOCK
       WRITE(OUTU,35) (BLCOEP(J), J=K, K+I-1)
       K=K+I
    end do
    !
    WRITE(OUTU,15) 'Matrix of BOND Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    do I=1,NBLOCK
       WRITE(OUTU,35) (BLCOEB(J), J=K, K+I-1)
       K=K+I
    end do
    !
    WRITE(OUTU,15) 'Matrix of ANGLE Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    do I=1,NBLOCK
       WRITE(OUTU,35) (BLCOEA(J), J=K, K+I-1)
       K=K+I
    end do
    !
    WRITE(OUTU,15) 'Matrix of DIHE Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    do I=1,NBLOCK
       WRITE(OUTU,35) (BLCOED(J), J=K, K+I-1)
       K=K+I
    end do
    !
    WRITE(OUTU,15) 'Matrix of CROSS Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    do I=1,NBLOCK
       WRITE(OUTU,35) (BLCOEC(J), J=K, K+I-1)
       K=K+I
    end do
    !
    WRITE(OUTU,15) 'Matrix of ELEC Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    do I=1,NBLOCK
       WRITE(OUTU,35) (BLCOEE(J), J=K, K+I-1)
       K=K+I
    end do
    !
    WRITE(OUTU,15) 'Matrix of VDW Interaction Coefficients'
    WRITE(OUTU,25)
    K=1
    IF(QBVSPLT) THEN
       WRITE(OUTU,15) 'Matrix of VDW Attractive Coefficients'
       WRITE(OUTU,25)
       K=1
       do I=1,NBLOCK
          WRITE(OUTU,35) (BLCOEVA(J), J=K, K+I-1)
          K=K+I
       end do

       WRITE(OUTU,15) 'Matrix of VDW Repulsive Coefficients'
       WRITE(OUTU,25)
       K=1
       do I=1,NBLOCK
          WRITE(OUTU,35) (BLCOEVR(J), J=K, K+I-1)
          K=K+I
       end do
    ELSE
       do I=1,NBLOCK
          WRITE(OUTU,35) (BLCOEV(J), J=K, K+I-1)
          K=K+I
       end do
    ENDIF
    !
    !
15  FORMAT(1X,A)
25  FORMAT(' ')
35  FORMAT(' ', 13F10.5)

  END subroutine pintmat
  !insert page break

#if KEY_DOCK==1
  !-----------------------------------------------------------------------
  SUBROUTINE PINTDOC(OUTU)
    !     PROCEDURE PRINT-ASYMMETRIC-INTERACTION-MATRIX
    !-----------------------------------------------------------------------
    use chm_kinds

    implicit none
    INTEGER, intent(in) :: OUTU
    INTEGER :: I,J,K, L
    !
    WRITE(OUTU,15)
    WRITE(OUTU,25)
    do I=1,NBLOCK
       K = (I-1)*NBLOCK + 1
       L = I*NBLOCK
       WRITE(OUTU,35) (BLDOCP(J), J=K, L)
    end do
    !
15  FORMAT(' Matrix of Asymetric Interaction Coefficients')
25  FORMAT(' ')
35  FORMAT(' ', 13F10.5)

  END subroutine pintdoc
#endif 
  !insert page break

  !-----------------------------------------------------------------------
  SUBROUTINE BLUPLST(ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX)
    !SBblock: Major overhaul by Stefan Boresch, Nov 1994
    !     This routine had been more or less obsolete as the
    !     Block postprocessing commands usually did not perform
    !     any list updates during reads from the trajectory.
    !     This could be made to work with setting CUTNB very large as
    !     long as there are no IMAGEs present.
    !     As BLOCK was modified to work with IMAGEs, the post-processing
    !     routines HAVE to do list updates.  It is recommended that
    !     this be done for every frame in the trajectory.
    !     The calling sequence is about the same as in UPDECI,
    !     (heurist.src) without support for a heuristic update scheme.
    !     It is also attempted to keep print out from update
    !     routines to a minimum (this ought to be possible in a
    !     more elegant way).
    !     One may consider replacing this routine by UPDECI, or
    !     using it just as a dummy routine to call UPDECI
    !SBb
    !
    !     PROCEDURE UPDATE-LISTS
    !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use stream
  use bases_fcm
  use coord
  use psf
  use hbondm
  use image
  use imgup, only: upimag0, imhbon
  use inbnd
  use param
  use energym
  use fast
  use ffieldm

    implicit none
    INTEGER ISTEP,IHBFRX,INBFRX,ILBFRX,IMGFRX

    LOGICAL QDONB,QDOIM
    INTEGER OLDPRL

    QDONB=.FALSE.
    QDOIM=.FALSE.

    !SBblock: Setting PRNLEV to 1 suppresses
    !SBb  printout; is restored to current value at end of routine
    OLDPRL=PRNLEV
    PRNLEV=1

    !SBblock: Trap negative values for INBFRQ or IMGFRQ and set
    !     both to 1 in this case.  Note: The user is allowed to
    !     set them to 0 (no updating), which most likely is
    !     stupid, but then, the user is allowed to do so many
    !     stupid things in CHARMM so that this is only "consistent"
    !SBb
    IF ((INBFRX < 0).OR.(IMGFRX < 0)) THEN
       INBFRX=1
       IMGFRX=1
       IF(PRNLEV >= 2) WRITE(OUTU,705)
    ENDIF

705 FORMAT(' MESSAGE FROM BLOCK: Heuristic updates are not '/ &
         ' supported.  Updates will be carried out for every  '/ &
         ' frame in the trajectory. This is the recommended   '/ &
         ' default. ')
    ! image list
    IF (NTRANS  >  0) THEN
       IF (IMGFRX  /=  0) THEN
          IF (MOD(ISTEP,IMGFRX)  ==  0 .AND. ISTEP  >  0) THEN
             QDOIM=.TRUE.
             CALL UPIMAG0(X, Y, Z, WMAIN, 0)
          ENDIF
       ENDIF
    ENDIF

    !     H-BOND LIST
    IF (IHBFRX  /=  0) THEN
       IF (MOD(ISTEP,IHBFRX)  ==  0) THEN
          CALL HBONDS(OUTU,IDON,IHD1,NDON,IACC,IAC1,NACC,IHB,JHB,KHB, &
               LHB,ICH,NHB,MAXHB,CHBA,CHBB,HBEXPN,CTONHB,CTOFHB,CTONHA, &
               CTOFHA,CUTHB,CUTHBA,LHBFG,NCH,KCH,IAC,ATC,NATC,IMOVE, &
               BEST,HBEXCL,BNBND%IBLO14, &
               BNBND%INB14,NNB14,X,Y,Z,NATOM)
          !
          !     IMAGE H-BOND LIST
          IF (NTRANS  >  0) CALL IMHBON(BIMAG,X,Y,Z)
       ENDIF
    ENDIF
    !
    !     NONBOND LIST
    IF (INBFRX  /=  0) THEN
       IF (MOD(ISTEP,INBFRX)  ==  0) THEN
          QDONB=.TRUE.
          CALL NBONDS(X,Y,Z,BNBND,BIMAG)
       ENDIF
    ENDIF

    !SBblock: No clue what that was supposed to be...
    !     LANGEVIN DYNAMICS
    !     IF (ILANG  ==  1) THEN
    !     IF (ISTEP  ==  0) THEN
    !     CALL LNGFIL(ILANG,GAMMA,RFD,TBATH,DELTA,RBUF,SXREF,
    !    1            SYREF,SZREF,X,Y,Z
#if KEY_ACE==1
    !    2           ,0,0,.FALSE.        
#endif
    !    3            )
    !     ELSE
    !     IF (ILBFRX  ==  0) THEN
    !     IF (MOD(ISTEP,ILBFRX)  ==  0) THEN
    !     CALL LNGFIL(ILANG,GAMMA,RFD,TBATH,DELTA,RBUF,SXREF,
    !    1            SYREF,SZREF,X,Y,Z
#if KEY_ACE==1
    !    2           ,0,0,.FALSE.        
#endif
    !    3            )
    !     ENDIF
    !     ENDIF
    !     ENDIF
    !     ENDIF
    !
    !     CHECK CONSISTENCY OF NON-BONDED DATA STRUCTURE
    if (.not. associated(BNBND%INBLO)) then
       call wrndie(-3, 'BLOCK', 'Non-bonded data structure is undefined')
    else if (size(BNBND%INBLO) < NATOM) then
       call wrndie(-3, 'BLOCK', 'Non-bonded data structure is too small')
    endif

    PRNLEV=OLDPRL

  END subroutine bluplst

  !-----------------------------------------------------------------------
  SUBROUTINE BVSPLTCK(BLCOVA,BLCOVR)
    !     This subroutine checks if the vdw block of scaling factors
    !     has been split into an attractive and repulsive part.
    !     It does this by comparing the elements in BLCOEVA and BLCOEVR,
    !     if all elements are the same then the VDW block has not been split,
    !     however if the elements are different then the VDW block
    !     has been split and the attractive and repulsive terms in the LJ vdw
    !     interaction will be scaled independently.
    !-----------------------------------------------------------------------
  use chm_kinds

    implicit none
    real(chm_real), dimension(ninter) :: BLCOVA, BLCOVR
    INTEGER :: I,N

    N = (NBLOCK*(NBLOCK+1))/2
    QBVSPLT=.false.
    do I=1,N
       IF( BLCOVA(I)  /=  BLCOVR(I) ) THEN
          QBVSPLT=.true.
       ENDIF
    end do

  END subroutine bvspltck

  !insert page break
  !-----------------------------------------------------------------------
  !.ab.Insert HybH routines here.
  !.ab.
  !-----------------------------------------------------------------------
  !.ab...Sum terms....
  SUBROUTINE SUMHYB(TERM,ER,EP)
    !...This routine puts the term where it belongs (once per proc...).
    !-----------------------------------------------------------------------
  use chm_kinds
  use energym
    implicit none
    INTEGER TERM
    real(chm_real) ER,EP
    !.ab.
    ETERMR(TERM)=ETERMR(TERM)+ER
    ETERMP(TERM)=ETERMP(TERM)+EP
    !.ab.
  END subroutine sumhyb
  !.ab.
  !-----------------------------------------------------------------------
  !.ab...Print terms....
  SUBROUTINE PRHYBH()
    !...This routine print out the differencial.
    !-----------------------------------------------------------------------
  use chm_kinds
  use energym
    implicit none
    !
    IF((QHYBH).AND.(OUTH > 0)) THEN
       WRITE(OUTH,2485) 'R',HYBHLB,ETERMR(DIHE),ETERMR(IMDIHE), &
            ETERMR(VDW),ETERMR(ELEC),ETERMR(EWKSUM),ETERMR(EWSELF), &
            ETERMR(EWEXCL)+ETERMR(EWQCOR)+ETERMR(EWUTIL)
       WRITE(OUTH,2485) 'P',HYBHLB,ETERMP(DIHE),ETERMP(IMDIHE), &
            ETERMP(VDW),ETERMP(ELEC),ETERMP(EWKSUM),ETERMP(EWSELF), &
            ETERMP(EWEXCL)+ETERMP(EWQCOR)+ETERMP(EWUTIL)
    ENDIF
2485 FORMAT(a1,1x,f6.4,7(1x,1pg24.16e2))
    !.ab.
  END subroutine prhybh
  !.ab.
  !-----------------------------------------------------------------------
  SUBROUTINE TESTHYBH(DL)
    !...This tests the derivatives of the forces.
    !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use number
  use energym
  use dimens_fcm
  use coord
  use deriv
  use bases_fcm
  use stream
  use comand
    implicit none
    real(chm_real) DL
    !
    INTEGER I
    real(chm_real) KEEPHL,EBEFOR(LENENT),EAFTER(LENENT)
    !
    !        1         2         3         4         5         6         7
    !23456789012345678901234567890123456789012345678901234567890123456789012
    CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
         .TRUE.,.TRUE.,.TRUE.,.TRUE., &
         0,[zero],[zero],[zero],[zero],[zero],[zero])
    KEEPHL=HYBHLB
    HYBHLB=KEEPHL-DL
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
    IF(PRNLEV >= 2) WRITE(OUTU,3496)
    IF(PRNLEV >= 2) WRITE(6,3472) &
         ' CALLING ENERGY WITH H-LAMBDA = ',HYBHLB
3472 FORMAT(A32,1pg24.16e2)
    DO I=1,LENENT
       EBEFOR(I)=ETERM(I)
    ENDDO
    HYBHLB=KEEPHL+DL
    IF(PRNLEV >= 2) WRITE(OUTU,3496)
    IF(PRNLEV >= 2) WRITE(6,3472) &
         ' CALLING ENERGY WITH H-LAMBDA = ',HYBHLB
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
    DO I=1,LENENT
       EAFTER(I)=ETERM(I)
    ENDDO
    HYBHLB=KEEPHL
    IF(PRNLEV >= 2) WRITE(OUTU,3496)
    IF(PRNLEV >= 2) WRITE(6,3472) &
         ' CALLING ENERGY WITH H-LAMBDA = ',HYBHLB 
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,3496)
       WRITE(OUTU,'(A)') 'TERM ITERM E-BEFOR E-AFTER DiffE/DL'
       WRITE(OUTU,'(A)') '            DEr/DL  DEp/DL    DE/DL'
       WRITE(OUTU,3496)
       DO I=1,LENENT
          IF((EBEFOR(I) /= 0.).AND.(EAFTER(I) /= 0.)) THEN
             WRITE(OUTU,3492) CETERM(I),I,EBEFOR(I),EAFTER(I), &
                  (EAFTER(I)-EBEFOR(I))/(DL+DL)
             WRITE(OUTU,3494)ETERMR(I),ETERMP(I),ETERMR(I)+ETERMP(I)
             WRITE(OUTU,3496)
          ENDIF
       ENDDO
    ENDIF

3492 FORMAT(1X,A4,1X,I3,3(1x,1pg17.09e2))
3494 FORMAT(9X,3(1x,1pg17.09e2))
3496 FORMAT('-----------------------------------------------------')

  END subroutine testhybh

  !-----------------------------------------------------------------------
  !C.ab...Set term....Moved out of the block region...
  !      SUBROUTINE SETTERM(TERM)
  !C...This routine flags the term used in HYBH.
  !C...to avoid incorporation of block.f90 elsewere.
  !-----------------------------------------------------------------------
  !...!#USE block_fcm
  !      INTEGER TERM
  !C
  !      IHYBH=TERM
  !C.ab.
  !      END subroutine setterm
  !C
  !-----------------------------------------------------------------------
  !.ab...Get zero radius.
  SUBROUTINE GETR02(R02,CZZ,A,B,IDXM,NDXM,I,J)
    !...This routine finds the root of the vdW+elec potential
    !...The elec is made attractive in case of repulsion to
    !...keep symmetry beteen the two. Returns the square...
    !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use stream
    implicit none
    !
    real(chm_real) R02,CZZ,A,B
    INTEGER IDXM,NDXM,I,J
    !
    real(chm_real) R0,DR,C,R6,R11
    INTEGER,save :: IKEEP=0
    !
    IF((IOLEV >= 0).AND.(IKEEP == 0)) then
       if (prnlev >= 2) WRITE(OUTU,'(i5,a20)') IDXM,' Atom types for HYBH'
    endif
    IKEEP=1
    !
    R0=(A/B)**SIXTH
    IF (CZZ /= ZERO) THEN
       C=-ABS(CZZ)
       !..find the root by iterations.
       R0=R0/TWO
10     CONTINUE
       R6=R0*R0*R0
       R6=R6*R6
       R11=R6*R6/R0
       DR=R0*(C*R11+A-B*R6)/(C*R11+TWELVE*A-SIX*B*R6)
       R0=R0+DR
       IF (ABS(DR) > TENM14) GOTO 10
    ENDIF
    !      write(6,'(a,5g)') 'R0',R0,CZZ,C,A,B
    R02=R0*R0
    if (prnlev >= 2) WRITE(OUTU,'(i5,a,f12.5,2i7)') NDXM,' HybH Pair R02: ',R02,I,J
    !.ab.
  END subroutine getr02
  !
  !-------------------------------------------------------------------------------------
  !.ab...INIT...
  SUBROUTINE INITHYBH()
    !...This routine inits arrays used in HYBH.
    !...to avoid incorporation of tunk.f90 elsewere.
    !-------------------------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use trunk
  use energym

    implicit none
    INTEGER :: I
    !.ab.Checks
    IF ((ELEC /= (VDW+1)).OR.(IMELEC /= (IMVDW+1))) CALL WRNDIE(-6, &
         '<INITHYBH>','ELEC=VDW+1 is required: check code (ENBxxx) !')
    IF (EWQCOR /= 47) CALL WRNDIE(-6,'<INITHYBH>', &
         'EWQCOR has changed, check code (ENBOND) !')
    IF (EWEXCL /= 46) CALL WRNDIE(-6,'<INITHYBH>', &
         'EWEXCL has changed, check code (ENBOND) !')
    !.ab.
    !.ab.Tables for truncation. Update scheme relies on primary image conservation.
    PTABLE=-1

    ! yw move call allocate_trunk from iniall.src
    ! ab put deallocate_trunk into the appropriate place
    ! also need maxaim to some proper value, natom?
    call allocate_trunk(maxaim)

    DO I=1,MAXAIM
       IDX(I)=-1
    ENDDO
    DO I=1,MAXTAB
       R02L(I)=-1.
    ENDDO
    !.ab.Note: it is important to leave R02L(1)=-1. -> first used index is two.
    NDXM=1
    DO I=1,MAXPAI
       NDX(I)=1
    ENDDO
    !.ab.
  END subroutine inithybh
  !-------------------------------------------------------------------------------------
  !.ab.Clear.HYBH....
  SUBROUTINE CLEARHH()
    !...This routine clears HYBH.
    !...to avoid incorporation of tunk.f90 elsewere.
    !-------------------------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use trunk
    implicit none
    call deallocate_trunk(maxaim)
    qhybh=.false.
  END subroutine clearhh

  !-------------------------------------------------------------------------------------
  !.ab...Printout...dEreac/dlambda,dEprod/dlambda,
  SUBROUTINE HYBHPR(OUTU)
    !...This routine returns the output freq/unit.
    !...to avoid incorporation of block.f90 elsewere.
    !-------------------------------------------------------------------------------------
    use chm_kinds
    use energym
    use number
    implicit none
    INTEGER, intent(in) :: OUTU
    !
    CALL PRINTE(OUTU,EPROPR,ETERMR,'HYBR','REA',.TRUE.,0,ZERO,ZERO,.FALSE.)
    CALL PRINTE(OUTU,EPROPP,ETERMP,'HYBP','PRO',.TRUE.,0,ZERO,ZERO,.FALSE.)
    !.ab.
  END subroutine hybhpr

  !-------------------------------------------------------------------------------------
  !.ab...Printout...forces ???
  !jlk---what does this subroutine do???

  SUBROUTINE PRFORC(OUTU)
    !...This routine returns the output freq/unit.
    !...to avoid incorporation of block.f90 elsewere.
    !-------------------------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord
  use ctitla
  use psf
  use shake
  use memory

    implicit none
    INTEGER, intent(in) :: OUTU
    real(chm_real),allocatable,dimension(:) :: IX
    real(chm_real),allocatable,dimension(:) :: IY
    real(chm_real),allocatable,dimension(:) :: IZ
    real(chm_real),allocatable,dimension(:) :: IXR
    real(chm_real),allocatable,dimension(:) :: IYR
    real(chm_real),allocatable,dimension(:) :: IZR
    !
    INTEGER :: I
    integer, dimension(maxaim) :: JUNK
    !
    call chmalloc('block_ltm.src','PRFORC','IX',NATOM,crl=IX)
    call chmalloc('block_ltm.src','PRFORC','IY',NATOM,crl=IY)
    call chmalloc('block_ltm.src','PRFORC','IZ',NATOM,crl=IZ)
    call chmalloc('block_ltm.src','PRFORC','IXR',NATOM,crl=IXR)
    call chmalloc('block_ltm.src','PRFORC','IYR',NATOM,crl=IYR)
    call chmalloc('block_ltm.src','PRFORC','IZR',NATOM,crl=IZR)

    IX(1:NATOM) = X(1:NATOM)
    IY(1:NATOM) = Y(1:NATOM)
    IZ(1:NATOM) = Z(1:NATOM)
    IXR(1:NATOM) = X(1:NATOM)
    IYR(1:NATOM) = Y(1:NATOM)
    IZR(1:NATOM) = Z(1:NATOM)

    X(1:NATOM) = IX(1:NATOM)
    Y(1:NATOM) = IY(1:NATOM)
    Z(1:NATOM) = IZ(1:NATOM)
    call chmdealloc('block_ltm.src','PRFORC','IX',NATOM,crl=IX)
    call chmdealloc('block_ltm.src','PRFORC','IY',NATOM,crl=IY)
    call chmdealloc('block_ltm.src','PRFORC','IZ',NATOM,crl=IZ)
    call chmdealloc('block_ltm.src','PRFORC','IXR',NATOM,crl=IXR)
    call chmdealloc('block_ltm.src','PRFORC','IYR',NATOM,crl=IYR)
    call chmdealloc('block_ltm.src','PRFORC','IZR',NATOM,crl=IZR)

    DO I=1,MAXAIM
       JUNK(I)=1
    ENDDO
    !      CALL CWRITE(OUTU,TITLEB,NTITLB,0,DXH,DYH,DZH,WMAIN,
    !     $     RES,TYPE,IBASE,NRES,NATOM,JUNK,0,2)
    !     $     RES,TYPE,IBASE,NRES,NATOM,JUNK,0,2)
    !.ab.
  END subroutine prforc

  !-----------------------------------------------------------------------
  !
  !.ab.End of HybH routines here.
  !
  !-----------------------------------------------------------------------
  !insert page break

  !-----------------------------------------------------------------------
  SUBROUTINE BLASGN(COMLYN,COMLEN,ITEMP,NATOM,IBLOCK,INDEX)
    !     THIS ROUTINE HANDLES THE SELECTION OF A SET OF ATOMS TO BE
    !     ASSIGNED AS BELONGING TO A SINGLE BLOCK; THAT BLOCK IS IDENTIFIED
    !     BY INDEX.
    !
    !yw...11-Aug-93, moved out of BLOCK code as it is used in other modules.
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use coord
    use select
    use stream
    implicit none
    INTEGER ITEMP(*), IBLOCK(*)
    INTEGER NATOM, COMLEN, INDEX
    CHARACTER(len=*) COMLYN
    !     LOCAL
    INTEGER I
    !
    !     BEGIN
    CALL SELCTA(COMLYN,COMLEN,ITEMP,X,Y,Z,WMAIN,.TRUE.)
    DO I=1, NATOM
       IF (ITEMP(I)  ==  1) THEN
          IF (IBLOCK(I) /= 1 .AND. WRNLEV >= 2) WRITE(OUTU,20) &
               I, IBLOCK(I), INDEX
          IBLOCK(I)=INDEX
       ENDIF
    ENDDO
20  FORMAT('WARNING: Atom number ', I5, &
         ' has been reassigned from block',I4,' to block',I4,'.')
    !
!-- ##ENDIF (block_fcm) ! IF BLOCK

  END subroutine blasgn
!(ldm)

  !insert page break
  !-----------------------------------------------------------------------
  SUBROUTINE LDMATRIX(QLDM, NBLOCK, ninter, BXLAMB, BLCOP, &
       BLCOB,BLCOA,BLCOD,BLCOE,BLCOV, lstrt)
    !-----------------------------------------------------------------------
  use chm_kinds

    implicit none
    integer, intent(in) :: nblock, lstrt, ninter
    integer :: i, j
    LOGICAL, intent(in) :: QLDM
    real(chm_real) :: lnorm
    real(chm_real), dimension(nblock), intent(inout) :: BXLAMB
    real(chm_real), dimension(ninter), intent(inout) :: BLCOP, BLCOB, BLCOA, BLCOD, BLCOE, BLCOV

    IF(QLDM) THEN
       !        TO ENSURE SUM LAMBDA**2 = 1
       LNORM = 0.0
       DO I = LSTRT, NBLOCK  
          LNORM = LNORM + BXLAMB(I)*BXLAMB(I)
       ENDDO
       IF(LNORM /= 1.0) THEN
          LNORM = SQRT(LNORM)
          DO I = LSTRT, NBLOCK 
             BXLAMB(I)=BXLAMB(I)/LNORM
          ENDDO
       ENDIF
       !        TRANSFORM TO BLCOP
       !         NINT=NBLOCK*(NBLOCK+1)/2
       !        Fisrt set all BLCOP = 0.0
       DO I = 1, ninter 
          BLCOP(I) = 0.0
          BLCOB(I) = 0.0
          BLCOA(I) = 0.0
          BLCOD(I) = 0.0
          BLCOE(I) = 0.0
          BLCOV(I) = 0.0
       ENDDO
       !        Conversion from lambdas to coefficient matrix
       !        NOTE THAT WE USE LAMBDA**2 INSTEAD OF LAMBDA ITSELF
       BLCOP(1) = BXLAMB(1)*BXLAMB(1)
       BLCOB(1) = BLCOP(1)
       BLCOA(1) = BLCOP(1)
       BLCOD(1) = BLCOP(1)
       BLCOE(1) = BLCOP(1)
       BLCOV(1) = BLCOP(1)
       DO I = LSTRT, NBLOCK 
          !:          First row elements
          J = 1 + I*(I-1)/2
          BLCOP(J) = BXLAMB(I)*BXLAMB(I)
          BLCOB(J) = BLCOP(J)
          BLCOA(J) = BLCOP(J)
          BLCOD(J) = BLCOP(J)
          BLCOE(J) = BLCOP(J)
          BLCOV(J) = BLCOP(J)
          !:          Diagonal elements
          J = I + I*(I-1)/2
          BLCOP(J) = BXLAMB(I)*BXLAMB(I)
          BLCOB(J) = BLCOP(J)
          BLCOA(J) = BLCOP(J)
          BLCOD(J) = BLCOP(J)
          BLCOE(J) = BLCOP(J)
          BLCOV(J) = BLCOP(J)
       ENDDO
    END IF

  END subroutine ldmatrix

  !---------------------------------------------------------------------
  SUBROUTINE RMLAMBDA
    !         invoked from keyword "RMLA" in BLOCK
    !---------------------------------------------------------------------

  use chm_kinds
  use dimens_fcm
  use string
  use stream
  use string
  use comand
    implicit none
    character(len=4) :: wd
    logical :: go, qerr
    integer :: count

    COUNT = 0
    GO = .TRUE.
    qerr = .false.
    do
       WD = NEXTA4(COMLYN,COMLEN)
       IF (STRLNG(WD) == 0) THEN
          GO = .FALSE.
       ELSE IF (WD == 'BOND') THEN
          QNOBO = .TRUE.
          QNOUB = .TRUE.
       ELSE IF (WD == '12BO') THEN
          QNOBO = .TRUE.
       ELSE IF (WD == '13BO') THEN
          QNOUB = .TRUE.
       ELSE IF (WD == 'THET'.OR.WD == 'ANGL') THEN
          QNOAN = .TRUE.
       ELSE IF (WD == 'PHI '.OR.WD == 'DIHE') THEN
          QNOPH = .TRUE.
       ELSE IF (WD == 'IMPH'.OR.WD == 'IMPR') THEN
          QNOIM = .TRUE.
       ELSE IF (WD == 'CMAP') THEN
          QNOCT = .TRUE.
       ELSE
          QNOBO = .FALSE.
          QNOUB = .FALSE.
          QNOAN = .FALSE.
          QNOPH = .FALSE.
          QNOIM = .FALSE.
          QNOCT = .FALSE.

          IF(WRNLEV >= 2) WRITE(OUTU,'(1X,A,A)') &
               'RMLAMBDA error with command: ',WD
          CALL WRNDIE(0,'<RMLAMDA>', &
               'Illegal energy option in RMLA command. Try again')
          IF(WRNLEV >= 2) WRITE(OUTU,*) &
               'All RMLAMBDA switches set to false.'
          GO = .FALSE.
          qerr = .true.
       END IF
       if (.not. go) exit
       COUNT = COUNT + 1
    enddo

    if (.not. qerr) then
       IF (COUNT > 0) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,*) 'RMLAMBDA list'
          IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,L1,A,L1,A,L1,A,L1,A,L1,A,L1)') &
               ' BOND ',QNOBO,' THETA ',QNOAN,' PHI ',QNOPH, &
               ' IMPHI ',QNOIM,' CMAP',QNOCT,' UREY-B',QNOUB
       ELSE
          CALL WRNDIE(0,'<RMLAMBDA>',' Empty option list')
       END IF
    endif

  END subroutine rmlambda
!ENDIF(ldm)

  !.ab...Set term....Moved out of the block region...
  SUBROUTINE SETTERM(TERM)
    !...This routine flags the term used in HYBH.
    !...to avoid incorporation of block.f90 elsewere.
  use chm_kinds
    !#USE block_fcm
    implicit none
    INTEGER TERM,TMP
    !.ab.TMP to fool compiler IFN BLOCK....
#if KEY_BLOCK==1
    IHYBH=TERM                 
#endif
    TMP=TERM
    !.ab.
  END subroutine setterm

  !insert page break
  !============================================================================= 
  !  new subroutines from jlk
  !============================================================================= 
  !---------------------------------------------------------------------------
#if KEY_SCCDFTB==1 /*sccdftb*/
  subroutine block_sccd(comlyn,comlen)
    !         invoked by BLOCK subcommand "SCCD"
    !---------------------------------------------------------------------------
  use string
  use number
  use blockscc_fcm 

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen

    qsccb=.true.
    qstop=(INDXA(COMLYN,COMLEN,'STOP') > 0)
    qdtop=(INDXA(COMLYN,COMLEN,'DTOP') > 0)
    qpkac=(INDXA(COMLYN,COMLEN,'PKAC') > 0)
    if(qpkac) then
       idxstp=GTRMI(COMLYN,COMLEN,'ISTP',1)
    endif

    !        write(outu,*) 'QSCCB = ',qsccb
    if(qdtop.and.(qpkac.eqv..false.)) then
       cdvdl(1)=zero
       cdvdl(2)=minone
       cdvdl(4)=one
       cdvdl(3)=minone
       cdvdl(5)=zero
       cdvdl(6)=one
    else if(qstop.and.(qpkac.eqv..false.)) then
       cdvdl(1)=zero
       cdvdl(2)=zero
       cdvdl(4)=zero
       cdvdl(3)=zero
       cdvdl(5)=zero
       cdvdl(6)=zero
    else if(qstop.and.qpkac.and.(idxstp == 1)) then
       cdvdl(1)=zero
       cdvdl(2)=zero
       cdvdl(4)=zero
       cdvdl(3)=zero
       cdvdl(5)=one
       cdvdl(6)=zero
    else if((idxstp == 2).and.qpkac) then
       cdvdl(1)=zero
       cdvdl(2)=zero
       cdvdl(4)=minone
       cdvdl(3)=zero
       cdvdl(5)=zero
       cdvdl(6)=zero
    endif

  END subroutine block_sccd

  !---------------------------------------------------------------------------
#endif /* (sccdftb)*/
  subroutine block_pssp(comlyn,comlen) 
    !         invoked by BLOCK subcommand "PSSP" 
    !---------------------------------------------------------------------------
    use number
    use sftc 
    use string
    use stream

    implicit none
    character(len=*), intent(inout) :: comlyn 
    integer, intent(inout) :: comlen 

    QBPSSP=.true.
    BALAMBD = GTRMF(COMLYN,COMLEN,'ALAM',FIVE) !for ELEC
    BDLAMBD = GTRMF(COMLYN,COMLEN,'DLAM',FIVE) !for VDW
    bdvdl(1)=zero
    bdvdl(2)=minone
    bdvdl(4)=one
    bdvdl(3)=minone
    bdvdl(5)=zero
    bdvdl(6)=one
    if (prnlev >= 2) write(outu,*)"Softcore ALAM(ELEC)/DLAM(VDW)=",BALAMBD,BDLAMBD

  END subroutine block_pssp

  !---------------------------------------------------------------------------
  subroutine block_nopssp 
    !         invoked by BLOCK subcommand
    !---------------------------------------------------------------------------
    use sftc
    use stream

    implicit none

    QBPSSP=.false.
    if (prnlev >= 2) write(outu,*)"Soft-core is removed. QBPSSP=",QBPSSP

  END subroutine block_nopssp

  !---------------------------------------------------------------------------
  subroutine block_coef(comlyn,comlen) 
    !         invoked by BLOCK subcommand "COEF"
    !---------------------------------------------------------------------------
  use chm_kinds
  use string
!%%AvdV-070721 bugfix (see below)
  use number, only: tenm8
!%%AvdV-070721--end

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer :: i, j, jint
    real(chm_real) :: r

!%%AvdV-070721 bugfix (see below)
    LOGICAL QVDWAR
!%%AvdV-070721--end   

    I=NEXTI(COMLYN,COMLEN)
    J=NEXTI(COMLYN,COMLEN)
    R=NEXTF(COMLYN,COMLEN)
    IF (I  >  J) THEN
       JINT=J
       J=I
       I=JINT
    ENDIF
    JINT=I+J*(J-1)/2
    BLCOEP(JINT) = R
    ! Now get the other block coefficients
    ! Set them equal to BLCOEP by default
    ! Tom Simonson, Oct. 95
    BLCOEB(JINT) = GTRMF(COMLYN,COMLEN,'BOND',R)
    BLCOEA(JINT) = GTRMF(COMLYN,COMLEN,'ANGL',R)
    BLCOED(JINT) = GTRMF(COMLYN,COMLEN,'DIHE',R)
    BLCOEC(JINT) = GTRMF(COMLYN,COMLEN,'CMAP',R)
    BLCOEE(JINT) = GTRMF(COMLYN,COMLEN,'ELEC',R)
    BLCOEV(JINT) = GTRMF(COMLYN,COMLEN,'VDW ',R)
!%%AvdV-070721 bugfix
!  fixes error: old code ignored when both vdwa and vdwr were set;       
!  e.g. coef 1 2 1.0 vdwa 0.0 vdwr 0.0 old code: QBVSPLT=.false.; blcov=1    
!                                      new code: QBVSPLT=.false.; blcov=0   
    IF((INDX(COMLYN,COMLEN,'VDWA',4).GT.0).AND.(INDX(COMLYN,COMLEN,'VDWR',4).GT.0))THEN
       QVDWAR=.TRUE.
    ELSE
       QVDWAR=.FALSE.
    ENDIF
    BLCOEVA(JINT) = GTRMF(COMLYN,COMLEN,'VDWA',BLCOEV(JINT))
    BLCOEVR(JINT) = GTRMF(COMLYN,COMLEN,'VDWR',BLCOEV(JINT))
    IF(QVDWAR.AND.(ABS(BLCOEVA(JINT)-BLCOEVR(JINT)).LT.TENM8))THEN
       BLCOEV(JINT)=BLCOEVA(JINT)
    ENDIF
!%%AvdV-070721--end

  END subroutine block_coef

  !---------------------------------------------------------------------------
#if KEY_DOCK==1 /*dock*/
  subroutine block_docf(comlyn,comlen)
    !         invoked by BLOCK subcommand "DOCF"
    !---------------------------------------------------------------------------
  use chm_kinds
  use string

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer :: i, j, jint
    real(chm_real) :: r

    IF (NBLOCK  >  0) QDOCK=.TRUE.
    ! Procedure SET-ASYMMETRIC COEFFICIENT
    I=NEXTI(COMLYN,COMLEN)
    J=NEXTI(COMLYN,COMLEN)
    R=NEXTF(COMLYN,COMLEN)
    R= 1.0/R
    JINT=(I-1)*NBLOCK + J
    BLDOCP(JINT) = R

  END subroutine block_docf
#endif /*  (dock) (DOCK)*/
  !---------------------------------------------------------------------------
  subroutine block_unsa 
    !         invoked by BLOCK subcommand "UNSA"
    !       turns off option to save energy components by block
    !---------------------------------------------------------------------------
  use chm_kinds
  use memory

    implicit none

    IF (QPRNTV) THEN
       call chmdealloc('block_ltm.src','block_unsa','VBBOND',NBLOCK,crl=VBBOND)
       call chmdealloc('block_ltm.src','block_unsa','VBANG',NBLOCK,crl=VBANG)
       call chmdealloc('block_ltm.src','block_unsa','VBTORS',NBLOCK,crl=VBTORS)
       call chmdealloc('block_ltm.src','block_unsa','VBIMPR',NBLOCK,crl=VBIMPR)
#if KEY_CMAP==1
       call chmdealloc('block_ltm.src','block_unsa','VBCMAP',NBLOCK,crl=VBCMAP)
#endif 
       call chmdealloc('block_ltm.src','block_unsa','VBGENB',NBLOCK,crl=VBGENB)
       call chmdealloc('block_ltm.src','block_unsa','VBELEC',NBLOCK,crl=VBELEC)
       call chmdealloc('block_ltm.src','block_unsa','VBVDW',NBLOCK,crl=VBVDW)
       QPRNTV = .FALSE.
    ENDIF

  END subroutine block_unsa

  !---------------------------------------------------------------------------
  subroutine block_save 
    !         invoked by BLOCK subcommand "SAVE"
    !       turns on option to save energy components by block
    !---------------------------------------------------------------------------
  use chm_kinds
  use memory

    implicit none

    IF (NBLOCK  >  0) QPRNTV = .TRUE.
    IF (QPRNTV)  THEN
       call chmalloc('block_ltm.src','block_save','VBBOND',NBLOCK,crl=VBBOND)
       call chmalloc('block_ltm.src','block_save','VBANG',NBLOCK,crl=VBANG)
       call chmalloc('block_ltm.src','block_save','VBTORS',NBLOCK,crl=VBTORS)
       call chmalloc('block_ltm.src','block_save','VBIMPR',NBLOCK,crl=VBIMPR)
#if KEY_CMAP==1
       call chmalloc('block_ltm.src','block_save','VBCMAP',NBLOCK,crl=VBCMAP)
#endif 
       call chmalloc('block_ltm.src','block_save','VBGENB',NBLOCK,crl=VBGENB)
       call chmalloc('block_ltm.src','block_save','VBELEC',NBLOCK,crl=VBELEC)
       call chmalloc('block_ltm.src','block_save','VBVDW',NBLOCK,crl=VBVDW)
    ENDIF

  END subroutine block_save

  !---------------------------------------------------------------------------
  subroutine block_call(comlyn,comlen,qcheck)
    !         invoked by BLOCK subcommand "CALL"
    !---------------------------------------------------------------------------
  use chm_kinds
  use memory
  use stream
  use string
  use psf, only : natom

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer :: i
    integer, allocatable, dimension(:) :: itemp
    logical, intent(out) :: qcheck

    !     Procedure ASSIGN-A-BLOCK
    I=NEXTI(COMLYN,COMLEN)
    IF (I  <=  NBLOCK) THEN
       call chmalloc('block_ltm.src','BLOCK','ITEMP',NATOM,intg=ITEMP)
       CALL BLASGN(COMLYN,COMLEN,ITEMP,NATOM,IBLCKP,I)
       call chmdealloc('block_ltm.src','BLOCK','ITEMP',NATOM,intg=ITEMP)
       QCHECK=.TRUE.
       IF(PRNLEV >= 2) WRITE(OUTU,30) I
30     FORMAT(' The selected atoms have been reassigned to block', I4)
    ELSE
       CALL WRNDIE(-3,'<BLOCK>', &
            'Failed attempt to reassign atoms.  Block number too high.')
    ENDIF

  END subroutine block_call

  !---------------------------------------------------------------------------
  subroutine block_lamb(comlyn,comlen) 
    !         invoked by BLOCK subcommand "LAMB"
    !---------------------------------------------------------------------------
  use chm_kinds
  use string

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    real(chm_real) :: r

    !     Procedure SET-COEFFICIENT-BY-LAMBDA
    R=NEXTF(COMLYN,COMLEN)
    IF (NBLOCK  /=  3) CALL WRNDIE(-3,'<BLOCK>', &
         'Three blocks must be used with this command')
    CALL LOADL(R)

  END subroutine block_lamb

  !---------------------------------------------------------------------------
  subroutine block_hybh(comlyn,comlen) 
    !         invoked by BLOCK subcommand "HYBH"
    !---------------------------------------------------------------------------
  use number
  use string

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen

    !     Procedure SET-COEFFICIENT-BY-LAMBDA
    HYBHLB=NEXTF(COMLYN,COMLEN)
    IF (NBLOCK  /=  3) CALL WRNDIE(-3,'<BLOCK>', &
         'Three blocks must be used with this command')
    IF (.NOT. QHYBH) CALL INITHYBH()
    QHYBH=.TRUE.
    !        HRR=ZERO
    !        HPR=ZERO
    CALL LOADL(ZERO)
    BLCOEP(4) = ONE
    BLCOEB(4) = ONE
    BLCOEA(4) = ONE
    BLCOED(4) = ONE
    BLCOEC(4) = ONE
    BLCOEV(4) = ONE
    BLCOEE(4) = ONE
    BLCOEP(6) = ONE
    BLCOEB(6) = ONE
    BLCOEA(6) = ONE
    BLCOED(6) = ONE
    BLCOEC(6) = ONE
    BLCOEV(6) = ONE
    BLCOEE(6) = ONE

    OUTH = -1

  END subroutine block_hybh

  !---------------------------------------------------------------------------
  subroutine block_end(comlyn,comlen,ltemp,qcheck,qend)
    !         invoked by BLOCK subcommand "END"
    !---------------------------------------------------------------------------
    use lambdam
    use stream
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_bonded_block,only:init_bonded_block
#endif
    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer, intent(in) :: ltemp
    logical, intent(in) :: qcheck
    logical, intent(out) :: qend

    !     Procedure FINISH-UP-AND-END
    CALL XTRANE(COMLYN,COMLEN,'BLOCK')
    IF (NBLOCK  >  0) THEN
       !         Check if the VDW energy term has been split, and
       !         declare the logical flag QBVSPLT
       IF (QBLOCK) CALL BVSPLTCK(BLCOEVA,BLCOEVR)
       IF (QCHECK) THEN
!ldm
          if (qmld .and. nsitemld.gt.2) then
             call msld_blchek(iblckp)
          else
! LDM
             CALL BLCHEK(IBLCKP)
          endif                                        !ldm
       ENDIF
       IF ((PRNLEV >= 2).AND.QBLOCK) CALL PINTMAT(OUTU)
#if KEY_DOCK==1
       IF ((PRNLEV >= 2).AND.QDOCK) CALL PINTDOC(OUTU) 
#endif
    ENDIF
!(ldm)
    IF (NRST == 2)THEN
       LSTRT=3
       CALL LDMATRIX(qldm, NBLOCK, ninter, BIXLAM, BLCOEP, &
            BLCOEB, BLCOEA, BLCOED, BLCOEE, BLCOEV,LSTRT)
       if (prnlev >= 2) WRITE (OUTU,200)
200    FORMAT('>>> BLOCK MATRIX WAS REBUILT ')
       IF ((PRNLEV >= 2).AND.QBLOCK) CALL PINTMAT(OUTU)
    ENDIF
    call ldm_mc_endblock(nblock,ltemp)
    if (qmld) call msld_assign_flags(iblckp,qnobo,qnoub,qnoan,qnoph,qnoim,qnoct)
!ENDIF (ldm) ! LDM

#if KEY_DOMDEC==1
    ! The block setup is done, now we can call domdec to initialize its block
    if (qmld .and. q_domdec) then
       call init_bonded_block()
    endif
#endif

    qend = .true.

  END subroutine block_end

  !---------------------------------------------------------------------------
  subroutine block_clear 
    !     invoked by BLOCK subcommand "CLEA"
    !---------------------------------------------------------------------------
    use chm_kinds
    use memory
    use gb_common,only: igentype,alph_gb,sigx_gb,sigy_gb,sigz_gb,t_gb  
    use psf, only : natom
    use dimens_fcm, only : maxaim
    use stream
    use lambdam
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_bonded_block,only:uninit_bonded_block
#endif
    implicit none
    integer :: nint, ndoc
    integer :: alloc_err

    !     Procedure CLEAR-BLOCKING
#if KEY_DOCK==1
    NDOC = NBLOCK*NBLOCK
    call chmdealloc('block_ltm.src','block_clear','BLDOCP',NDOC,crl=BLDOCP)
#endif 
    NINT=NBLOCK*(NBLOCK+1)/2
    call chmdealloc('block_ltm.src','block_clear','BLCOEP',NINT,crl=BLCOEP)
    ! clear the arrays for the individual energy terms (T.S., 10/95)
    call chmdealloc('block_ltm.src','block_clear','BLCOEB',NINT,crl=BLCOEB)
    call chmdealloc('block_ltm.src','block_clear','BLCOEA',NINT,crl=BLCOEA)
    call chmdealloc('block_ltm.src','block_clear','BLCOED',NINT,crl=BLCOED)
    call chmdealloc('block_ltm.src','block_clear','BLCOEC',NINT,crl=BLCOEC)
    call chmdealloc('block_ltm.src','block_clear','BLCOEE',NINT,crl=BLCOEE)
    call chmdealloc('block_ltm.src','block_clear','BLCOEV',NINT,crl=BLCOEV)
    call chmdealloc('block_ltm.src','block_clear','BLCOEVR',NINT,crl=BLCOEVR)
    call chmdealloc('block_ltm.src','block_clear','BLCOEVA',NINT,crl=BLCOEVA)
    ! clear the arrays for the block interaction energies (T.S., 3/96)
    call chmdealloc('block_ltm.src','block_clear','BLENEE',NINT,crl=BLENEE)
    call chmdealloc('block_ltm.src','block_clear','BLENEV',NINT,crl=BLENEV)
    IF(IGenType  ==  1) THEN
       deallocate(gb_block, stat=alloc_err)
       if(alloc_err /= 0) call wrndie(-4,"<block_ltm.src>BLOCK", &
            "C: Cannot deallocate  gb_block")

       call ldm_clear_gb(nblock,gb_lamb,gbldm)  !ldm
    ELSE IF (IGenType  ==  2) THEN
       deallocate(alph_gb,sigx_gb,sigy_gb,sigz_gb,t_gb,gb_block, &
            stat=alloc_err)
       if(alloc_err /= 0) call wrndie(-4,"<block_ltm.src>BLOCK", &
            "E: Cannot deallocate  gb arrays")
       allocate(alph_gb(natom),sigx_gb(natom),sigy_gb(natom), &
            sigz_gb(natom),t_gb(natom), &
            stat=alloc_err)
       if(alloc_err /= 0) call wrndie(-4,"<block_ltm.src>BLOCK", &
            "F: Cannot allocate memory for sigx_gb to gb_block")
    ENDIF
    IF (QPRNTV)  THEN
       call chmdealloc('block_ltm.src','block_clear','VBBOND',NBLOCK,crl=VBBOND)
       call chmdealloc('block_ltm.src','block_clear','VBANG',NBLOCK,crl=VBANG)
       call chmdealloc('block_ltm.src','block_clear','VBTORS',NBLOCK,crl=VBTORS)
       call chmdealloc('block_ltm.src','block_clear','VBIMPR',NBLOCK,crl=VBIMPR)
#if KEY_CMAP==1
       call chmdealloc('block_ltm.src','block_clear','VBCMAP',NBLOCK,crl=VBCMAP)
#endif 
       call chmdealloc('block_ltm.src','block_clear','VBGENB',NBLOCK,crl=VBGENB)
       call chmdealloc('block_ltm.src','block_clear','VBELEC',NBLOCK,crl=VBELEC)
       call chmdealloc('block_ltm.src','block_clear','VBVDW',NBLOCK,crl=VBVDW)
    ENDIF

    call ldm_block_clear(nblock,nreplica)   !ldm

#if KEY_DOMDEC==1
    if (q_domdec) then
       call uninit_bonded_block()
    endif
#endif

    !SBblock: should also remove the IBLOCK/IBLCKP data structure?!
    call chmdealloc('block_ltm.src','block_clear','IBLCKP',MAXAIM,intg=IBLCKP)
    !SBb
    QBLOCK=.FALSE.
#if KEY_DOCK==1
    QDOCK=.FALSE.
#endif /* DOCK*/
    NOFORC=.FALSE.
    QBFIX=.FALSE.
    QBDIS=.FALSE.
    QPRNTV = .FALSE.
    QNOBO = .FALSE.
    QNOUB = .FALSE.
    QNOAN = .FALSE.
    QNOIM = .FALSE.
    QNOPH = .FALSE.
    QNOCT = .FALSE.
    qsoft_omm = .false.
    IF(QHYBH)THEN
       CALL CLEARHH()
    ENDIF
    IF(PRNLEV >= 2) WRITE(OUTU,560)
560 FORMAT(' Block Structure has been turned off.')

  END subroutine block_clear

  !---------------------------------------------------------------------------
  subroutine block_bfix(comlyn,comlen)
    !        invoked by BLOCK subcommand "BFIX"
    !---------------------------------------------------------------------------
  use chm_kinds
  use memory
  use psf, only : natom
  use stream

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer, allocatable, dimension(:) :: itemp

    !     Procedure PROCESS-BFIX-COMMAND
    IF (QBFIX) THEN
       call chmdealloc('block_ltm.src','block_bfix','BPXFIX',BLNFIX,crl=BPXFIX)
       call chmdealloc('block_ltm.src','block_bfix','BPYFIX',BLNFIX,crl=BPYFIX)
       call chmdealloc('block_ltm.src','block_bfix','BPZFIX',BLNFIX,crl=BPZFIX)
       call chmdealloc('block_ltm.src','block_bfix','BPIFIX',BLNFIX,intg=BPIFIX)
       BLNFIX=0
       QBFIX=.FALSE.
    ENDIF
    call chmalloc('block_ltm.src','block_bfix','ITEMP',NATOM,intg=ITEMP)
    CALL BFIX1(COMLYN,COMLEN,ITEMP,NATOM)
    IF(PRNLEV >= 2) WRITE(OUTU,670)
670 FORMAT(' The selected atoms have been fixed in the main ', &
         'coordinate set.')
    call chmdealloc('block_ltm.src','block_bfix','ITEMP',NATOM,intg=ITEMP)

  END subroutine block_bfix

  !---------------------------------------------------------------------------
  subroutine block_bdis(comlyn,comlen)
    !        invoked by BLOCK subcommand "BDIS"
    !---------------------------------------------------------------------------
  use chm_kinds
  use memory
  use psf, only : natom
  use stream

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    integer, allocatable, dimension(:) :: itemp, jtemp

    !     PROCEDURE PROCESS-BDIS-COMMAND
    IF (QBDIS) THEN
       call chmdealloc('block_ltm.src','block_bdis','BPXDIS',BLNDIS,crl=BPXDIS)
       call chmdealloc('block_ltm.src','block_bdis','BPYDIS',BLNDIS,crl=BPYDIS)
       call chmdealloc('block_ltm.src','block_bdis','BPZDIS',BLNDIS,crl=BPZDIS)
       call chmdealloc('block_ltm.src','block_bdis','BPIDIS',BLNDIS,intg=BPIDIS)
       call chmdealloc('block_ltm.src','block_bdis','BPJDIS',BLNDIS,intg=BPJDIS)
       BLNDIS=0
       QBDIS=.FALSE.
    ENDIF
    call chmalloc('block_ltm.src','block_bdis','ITEMP',NATOM,intg=ITEMP)
    call chmalloc('block_ltm.src','block_bdis','JTEMP',NATOM,intg=JTEMP)
    call BDIS1(COMLYN,COMLEN,ITEMP,JTEMP,NATOM)
    call chmdealloc('block_ltm.src','block_bdis','JTEMP',NATOM,intg=JTEMP)
    call chmdealloc('block_ltm.src','block_bdis','ITEMP',NATOM,intg=ITEMP)

    IF(PRNLEV >= 2) WRITE(OUTU,680)
680 FORMAT(' The selected first set of atoms will be copied into ', &
         'the coordinates',/, &
         ' of the second set of atoms in analyzing trajectories.')

  END subroutine block_bdis
  !
  !
  subroutine set_block_exclusions(comlyn, comlen, qaddexcl)
    use chm_kinds
    use memory
    use psf, only : natom
    use stream
    use string

    implicit none
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen
    logical, intent(in) :: qaddexcl

    integer :: blockcnt(nblock) ! dynamic array to get atom count in each block
    integer, allocatable, dimension(:,:) :: atomsinBlocks ! allocatable array to provide list of atoms in each block
    integer :: maxatm_inBlock
    integer :: iblk, jblk, i, j

    blockcnt(1:nblock) = 0
    do i = 1, natom
       blockcnt(iblckp(i)) = blockcnt(iblckp(i)) + 1
    enddo
 
    maxatm_inBlock = 0
    do i = 1, nblock
       if(blockcnt(i)>maxatm_inBlock) maxatm_inBlock = blockcnt(i)
    enddo
    call chmalloc('block_ltm.src','set_block_exclusions','atomsinBlocks',nblock,maxatm_inBlock,intg=atomsinBlocks)
    blockcnt(1:nblock) = 0
    do i = 1, natom
       blockcnt(iblckp(i)) = blockcnt(iblckp(i)) + 1
       atomsinBlocks(iblckp(i),blockcnt(iblckp(i))) = i
    enddo

    if(.not. qaddexcl) nblock_excldPairs = 0
    do while (comlen > 0)
       ! Parse pair of blocks, iblk, jblk  
       iblk = nexti(comlyn,comlen)
       jblk = nexti(comlyn,comlen)
       if(iblk <= 0 .or. jblk <= 0) &
            call wrndie(-1, 'set_block_exclusions','IBLK or JBLK in excluded pair = 0')
       if(iblk > nblock .or. jblk > nblock) &
            call wrndie(-1, 'set_block_exclusions','IBLK or JBLK in excluded pair > NBLOCK')
     
       if(iblk <= nblock .and. jblk <= nblock .and. iblk * jblk > 0) then
          ! Allocate space to add iblk*jblk new exclusions
           if (allocated(block_excldPairs_I)) then
             call chmrealloc('block_ltm.src','set_block_exclusions','block_excldPairs_I', &
                  nblock_excldPairs+blockcnt(iblk)*blockcnt(jblk),intg=block_excldPairs_I)
             call chmrealloc('block_ltm.src','set_block_exclusions','block_excldPairs_J', &
                  nblock_excldPairs+blockcnt(iblk)*blockcnt(jblk),intg=block_excldPairs_J)
          else
             call chmalloc('block_ltm.src','set_block_exclusions','block_excldPairs_I', &
                  nblock_excldPairs+blockcnt(iblk)*blockcnt(jblk),intg=block_excldPairs_I)
             call chmalloc('block_ltm.src','set_block_exclusions','block_excldPairs_J', &
                  nblock_excldPairs+blockcnt(iblk)*blockcnt(jblk),intg=block_excldPairs_J)
          endif
          do i = 1, blockcnt(iblk)
             do j = 1, blockcnt(jblk)
                nblock_excldPairs = nblock_excldPairs + 1
                if(atomsinBlocks(iblk,i)<=atomsinBlocks(jblk,j)) then
                   block_excldPairs_I(nblock_excldPairs) = atomsinBlocks(iblk,i)
                   block_excldPairs_J(nblock_excldPairs) = atomsinBlocks(jblk,j)
                else
                   block_excldPairs_J(nblock_excldPairs) = atomsinBlocks(iblk,i)
                   block_excldPairs_I(nblock_excldPairs) = atomsinBlocks(jblk,j)
                endif
             enddo
          enddo
       endif
    enddo
    call chmdealloc('block_ltm.src','set_block_exclusions','atomsinBlocks', &
         nblock,maxatm_inBlock,intg=atomsinBlocks)
    qblock_excld_upinb = .true.
    qblock_excld = .true.
 
    if( prnlev >= 3 ) then
        WRITE(OUTU,*) 'Number of block exclusions nblock_excldPairs=', nblock_excldPairs
    endif

  end subroutine set_block_exclusions

  subroutine block_exclusions(ipk,jpk,npair)
    integer, intent(inout) :: ipk(*), jpk(*), npair

    integer :: i
    qblock_excld_upinb = .false.
    do i = 1, nblock_excldPairs

       npair = npair + 1
       ipk(npair) = block_excldPairs_I(i)
       jpk(npair) = block_excldPairs_J(i)

    enddo
  end subroutine block_exclusions

  subroutine block_clear_exclusions
    use chm_kinds
    use memory
    use bases_fcm, only: bnbnd
    implicit none
    integer :: asize, j
    character (len=4) winit
    if (allocated(block_excldPairs_I)) then
       asize = size(block_excldPairs_I)
       call chmdealloc('block_ltm.src','block_clear_exclusions','block_excldPairs_I', &
            asize,intg=block_excldPairs_I)
    endif
    if (allocated(block_excldPairs_J)) then
       asize = size(block_excldPairs_J)
       call chmdealloc('block_ltm.src','block_clear_exclusions','block_excldPairs_J', &
            asize,intg=block_excldPairs_J)
    endif
    qblock_excld_upinb = .true.
    qblock_excld = .false.
    winit = 'RESE'
    j=4
    call gtnbct(winit,j,bnbnd)

  end subroutine block_clear_exclusions
  !---------------------------------------------------------------------------

#endif /* (block_subs)*/

end module block_fcm
