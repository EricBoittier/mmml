module cheq

  use chm_kinds
  use chm_types
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="cheqmodule.src"
  !
  !     This file contains information for charge dynamics. The data structure
  !     is  part PSF type.
  !
  !     PSF type:
  !       ECH  --  electronegativity for atoms
  !       EHA  --  hardnss for atoms
  !       MOLT --  Molecular label
  !       MOLNT --  Total number of molecules
  !       MOLBL -- store atomic label in molcular order
  !       MOLPTR --  Pointer to MOLBL at the first atom of each molecule
  !                  MOLNT+1 Nonzeros. MOLPTR(MOLNT+1)=NATOM+1
  !       LAMDAQ --lagrangian constraint constant for electronegativity
  !                 equalization (Molecular)
  !       LAMDAQG --lagrangian constraint constant for electronegativity
  !                  equalization (Global)
  !       FQTOT -- Total charge, used in charge optimization
  !       QCGRP -- Take group as molecules in cg dynamics
  !
  !     DYNAMC:
  !       QMOLC  --  .TRUE. means charges are constrained to individual molecule
  !                  .FALSE. global charge flow is allowed
  !       MASSQ  --  fictitious charge mass used in charge dynamics
  !       MASSQD --  default value of fictitious charge mass used in charge dynamics
  !       IOCG -- Charge output flag, >=0 output along with trajectory
  !       TCG -- charge temperature, measure of cg kinetic energy
  !       TCGDEF -- default value for TCG, set to 5K at present
  !       QPOLAR1 -- true to polarize only one molecule
  !       IPOLAR1 -- the molecule to be polarized
  !       QFLUCQ  -- TRAJ of fluq stuff, CG being read or written
  !       CGEQ -- CG temperature contral, 
  !              == 0 zero CG temperature during equilibration
  !              == -1 do nothing to CG kinetic part
  !              > 0 thermostating !! to be developed, is currently zeroing
  !       QNOCO -- FIX ATOM FLUCQ DYNAMICS flag
  !       QCGWATER -- no force for water
  !
  !     ENERGY:
  !       QCG  --  .TRUE. means charge driving force being calculated
  !       QCGMINE  -- .TRUE. means 2nd derivative for charge optimization is
  !                   computed
  !       CGMODEL -- FORCE field used.
  !       QCGINV -- matrix inversion, CG consistent with conformation
  !       QCGINVF -- Charge force in matrix inversion
  !       QDCGN -- Charge force normalization
  !       QCGPOL -- Compute molecular polarizability tensor
  !     MINMIZ:
  !       CGFIX,CGTMP -- working space pointer
  !
  !     OTHER:
  !       MOLTYPE -- Atom flag:  -1 means rigid water
  !                               0 means flexable mol (default)
  !                               1 Berne SPC water
  !                  xyz force caused by flucq calculated only for 0 flag atoms
  !
  !  tom's new variables
  ! QPRTPTR  - pointer array into QPART; (I+1) holds end of Ith partition
  ! QPART    - holds atom numbers in partition order
  ! QPARTBIN - number of bin to which atom belongs
  ! QNPART   - number of partitions
  ! QCGSPC   - 
  ! QPRNETA  - print eta matrix from within CGINV if true
  ! QREF -- Use reference charges; fill from values in WMAIN
  ! CGREF -- array of reference charges
  !
  !

  private
  !---------------------------------------------------------------------
  !     PUBLIC things
  !
  !---ROUTINES-----
  public cpcgimg
#if KEY_CHEQ==1
  public cheqprep,cheq_iniall,cheqstop, &
       molsrt,molcgt,getvr1q,putvr1q, &
       checketa,checkqnorm,qcgset, &
       qbener,qcgtransi,gbfoncheq,qnorm,ddcg,qnocod, allocate_cheq, allocate_cheqinit &
#if KEY_PARALLEL==1
       ,cgcpy &                                         
#endif
       ;                    ! Ends continuation when not PARALLEL

  !---Variables----
  logical,save,public ::  qcg, &
       QCGPOL,QCGINVF,QMOLC, QPRNETA, QCGINV, &
       qnoco,qcgwater, qpolar1,cheqbasetup,qcgmine,gqt4p, &
       qdcgn,qcgrp,qpbeq1, minnorm, &
       qref

  integer,save,public :: &
       cgmodel,cgeq, iocg, ipolar1, qfrq, noseflag, molnt, &
       qnpart

  integer,allocatable,dimension(:),public ::  QPARTBIN,MOLT,MOLPTR, MOLBL

  integer,save,public :: CHEQNHS,CHEQNHSO,FQNHSWAT,FQNHSOWAT

  integer,save,public :: IFIRSTBATH(10),ILASTBATH(10), &
       NQBATHS,NDGFBATH(10)

  real(chm_real),save,public ::  &
       massq, tcg, massqd, tcgdef, cheqtemp, &
       KECGBATH(10),FQNHSBATH(10),FQNHSOBATH(10), &
       FQNHMBATH(10), &
       FQTEMPBATH(10)
  real(chm_real),allocatable,dimension(:),public :: ECH, EHA, CGREF

  real(chm_real),allocatable,dimension(:),save,public :: pmassq, &
       cgfix,cgtmp

  !
  !  DCH - Charge force
  !  SUMDCH - Sum of squared DCH
  !  QCG - true for charge optimization.
  !
  real(chm_real),public,allocatable,dimension(:) :: DCH,DDCH
  real(chm_real),public :: SUMDCH
  !
  !      End PUBLIC things
  !----------------------------------------------------------------------

  real(chm_real),save :: QRA1DEF,QRA2DEF,QRB1DEF,QRB2DEF, &
       QRQ1DEF,QRQ2DEF,QRKDEF

  real(chm_real),allocatable,dimension(:) :: LAMDAQ
  real(chm_real),save :: FQTOT, LAMDAQG, DEFQ

  integer,save,allocatable,dimension(:) ::  MOLTYPE

  logical,save ::  &
       QCGSPC, QT4P, &
       qhpwallp,QHPMASQ
  logical,save,allocatable,dimension(:) :: QPARTED

  integer,save,allocatable,dimension(:) :: QPRTPTR, QPART

  integer,save :: chEQallocCNTRMASQ

  real(chm_real),allocatable,dimension(:),save ::  &
       pwallpa1,pwallpb1,pwallpa2, &
       pwallpb2,pwallpQ1,pwallpQ2, &
       pwallpK

  integer,save :: wallpN

  integer,save :: PTYP,CHEQallocCNTRWALL

  integer :: alloc_err
#else /**/
  logical, parameter, public :: qcg = .false.
#endif 

  !=================================================
contains
  !=================================================
#if KEY_CHEQ==1 /*cheq_main*/
  subroutine allocate_cheq(natom,ngrp)
    use memory
    use replica_ltm
    use stream, only: outu, prnlev
    use psf, only: ib, jb, nbond, igpbs
    use number

    integer,allocatable,dimension(:,:) :: iwork
    integer natom, ngrp
    integer oldatm, oldgrp
    character(len=*),parameter :: routine_name="allocate_cheq"

    if (allocated(qpartbin)) then
       oldatm = size(qpartbin)
       oldgrp = size(lamdaq)
       if (oldatm == natom .and. oldgrp == ngrp) return

       if (oldatm < natom) then
          call chmrealloc(file_name,routine_name,'qpartbin',natom,intg=qpartbin)
          call chmrealloc(file_name,routine_name,'molptr',natom,intg=molptr)
          call chmrealloc(file_name,routine_name,'molbl',natom,intg=molbl)
          call chmrealloc(file_name,routine_name,'dch',natom,crl=dch)
          call chmrealloc(file_name,routine_name,'ddch',natom,crl=ddch)
          call chmrealloc(file_name,routine_name,'moltype',natom,intg=moltype)
          call chmrealloc(file_name,routine_name,'qparted',natom,log=qparted)
          call chmrealloc(file_name,routine_name,'qpart',natom,intg=qpart)

          qpartbin(oldatm+1:) = 0
          molptr(oldatm+1:) = 0
          molbl(oldatm+1:) = 0
          dch(oldatm+1:) = ZERO
          ddch(oldatm+1:) = ZERO
          moltype(oldatm+1:) = 0
          qparted(oldatm+1:) = .false.
          qpart(oldatm+1:) = 0
       endif

       if (oldgrp < ngrp) then
          call chmrealloc(file_name,routine_name,'lamdaq',ngrp,crl=lamdaq)
          call chmrealloc(file_name,routine_name,'qprtptr',ngrp+1,intg=qprtptr)

          lamdaq(oldgrp+1:) = ZERO
          qprtptr(oldgrp+2:) = 0
       endif
    else
       call chmalloc(file_name,routine_name,'qpartbin',natom,intg=qpartbin)
       call chmalloc(file_name,routine_name,'molptr',natom,intg=molptr)
       call chmalloc(file_name,routine_name,'molbl',natom,intg=molbl)
       call chmalloc(file_name,routine_name,'dch',natom,crl=dch)
       call chmalloc(file_name,routine_name,'ddch',natom,crl=ddch)
       call chmalloc(file_name,routine_name,'moltype',natom,intg=moltype)
       call chmalloc(file_name,routine_name,'qparted',natom,log=qparted)
       call chmalloc(file_name,routine_name,'qpart',natom,intg=qpart)

       call chmalloc(file_name,routine_name,'lamdaq',ngrp,crl=lamdaq)
       call chmalloc(file_name,routine_name,'qprtptr',ngrp+1,intg=qprtptr)

       qpartbin = 0
       molptr = 0
       molbl = 0
       dch = ZERO
       ddch = ZERO
       moltype = 0
       qparted = .false.
       qpart = 0
       lamdaq = ZERO
       qprtptr = 0
    endif

#if KEY_REPLICA==1
    IF(.NOT.QREP) THEN      
#endif
       !  find molecule label
       CALL MOLSRT(NATOM,IB,JB,NBOND,QCGRP,NGRP,IGPBS)
       IF (QCGRP.AND.PRNLEV.GE.2) WRITE(OUTU,'(A,/,A)') &
            'allocate_cheq> Treat GRP as Molecules in CG dynamics.', &
            '  This is true for the whole PSF.'
#if KEY_REPLICA==1
    ENDIF                   
#endif

    return
  end subroutine allocate_cheq

  subroutine allocate_cheqinit()
    use memory
    use number
    character(len=*),parameter :: routine_name="allocate_cheqinit"
#ifdef KEY_RESIZE
    integer:: i0
    i0=2
    call chmalloc(file_name,routine_name,'molt    ',i0,intg=molt)
    call chmalloc(file_name,routine_name,'ech     ',i0, crl=ech)
    call chmalloc(file_name,routine_name,'eha     ',i0, crl=eha)
#else
    call chmalloc(file_name,routine_name,'molt    ',maxaim,intg=molt)
    call chmalloc(file_name,routine_name,'ech     ',maxaim, crl=ech)
    call chmalloc(file_name,routine_name,'eha     ',maxaim, crl=eha)
#endif    
    molt = 1
    ech = zero
    eha = zero
    return
  end subroutine allocate_cheqinit

  subroutine cheq_iniall()

    integer :: i

    QMOLC=.TRUE.
    QCG=.FALSE.
    QCGMINE=.FALSE.
    QCGINV=.FALSE.
    QCGINVF=.FALSE.
    QPRNETA=.FALSE.
    QCGWATER=.FALSE.
    QDCGN=.TRUE.
    MOLNT=1
    !     MASSQD is in (ps/e)**2 kcal/mol
    MASSQD = 0.000065
    DEFQ = 6.9e-5
    !     WALL POTENTIAL FOR RESTRAINING CHARGES IN "NORMAL" RANGE OF MAGNITUDES
    QRA1DEF=1.500
    QRB1DEF=1.7500
    QRA2DEF=-1.50
    QRB2DEF=-1.7500
    QRQ1DEF=2.00
    QRQ2DEF=-2.000
    QRKDEF=1.0D-22
    !     The charge temperature is set to 5K
    TCGDEF=5.0
    !     Use orginal force field and lagrangian as default
    CGMODEL=0
    QPOLAR1=.FALSE.
    IPOLAR1=-1
    !     initialize counter for CHEQ keyword-based  ALLOCATION
    CHEQallocCNTRWALL=1
    CHEQallocCNTRMASQ=1
    QHPWALLP=.FALSE.
    QHPMASQ=.FALSE.

    return
  end subroutine cheq_iniall


  !-----------------------------------------------------------------------
  SUBROUTINE CHEQPREP(COMLYN,COMLEN)

    use memory
    use number
    use coord
    use stream
    use string
    use psf

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN

    LOGICAL   WALL
    LOGICAL   MASQ
    character(len=*), parameter :: proc_name = 'CHEQPREP'

    WALL=INDXA(COMLYN,COMLEN,'WALL') > 0

! Allocate space for system
    call allocate_cheq(natom,ngrp)

    IF (WALL .AND. CHEQallocCNTRWALL == 1) THEN
       call chmalloc(file_name,proc_name,'pwallpa1',NATOM,crl=pwallpa1)
       call chmalloc(file_name,proc_name,'pwallpb1',NATOM,crl=pwallpb1)
       call chmalloc(file_name,proc_name,'pwallpa2',NATOM,crl=pwallpa2)
       call chmalloc(file_name,proc_name,'pwallpb2',NATOM,crl=pwallpb2)
       call chmalloc(file_name,proc_name,'pwallpQ1',NATOM,crl=pwallpQ1)
       call chmalloc(file_name,proc_name,'pwallpQ2',NATOM,crl=pwallpQ2)
       call chmalloc(file_name,proc_name,'pwallpK',NATOM,crl=pwallpK)
       pwallpQ1 = ZERO
       pwallpQ2 = ZERO

       CHEQallocCNTRWALL = CHEQallocCNTRWALL + 1    ! ONE-TIME ASSIGNMENT ONLY
       QHPWALLP=.TRUE.
    ENDIF


    IF (CHEQallocCNTRMASQ == 1) THEN
       call chmalloc(file_name,proc_name,'pmassq',NATOM,crl=pmassq)

       CHEQallocCNTRMASQ  = CHEQallocCNTRMASQ + 1
       QHPMASQ=.TRUE.
    ENDIF

    CALL FQCOM(COMLYN,COMLEN)

    RETURN
  END SUBROUTINE CHEQPREP

  !-----------------------------------------------------------------------
  SUBROUTINE CHEQSTOP
    use memory
    use psf
    character(len=*), parameter :: proc_name = 'CHEQSTOP'

    IF (QHPWALLP .and. allocated(pwallpa1)) THEN
       call chmdealloc(file_name,proc_name,'pwallpa1',NATOM,crl=pwallpa1)
       call chmdealloc(file_name,proc_name,'pwallpb1',NATOM,crl=pwallpb1)
       call chmdealloc(file_name,proc_name,'pwallpa2',NATOM,crl=pwallpa2)
       call chmdealloc(file_name,proc_name,'pwallpb2',NATOM,crl=pwallpb2)
       call chmdealloc(file_name,proc_name,'pwallpQ1',NATOM,crl=pwallpQ1)
       call chmdealloc(file_name,proc_name,'pwallpQ2',NATOM,crl=pwallpQ2)
       call chmdealloc(file_name,proc_name,'pwallpK',NATOM,crl=pwallpK)
    ENDIF

    IF (QHPMASQ) THEN
       call chmdealloc(file_name,proc_name,'pmassq',NATOM,crl=pmassq)
    ENDIF

    RETURN
  END SUBROUTINE CHEQSTOP

  !------------------------------------------------------------------------------
  SUBROUTINE FQCOM(COMLYN,COMLEN)
    ! This subroutine is called from CHARMM (main) to process FLUQ commands
    !
    ! written by Tom Cleveland
    use number
    use coord
    use select
    use stream
    use string
    use psf
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    INTEGER FLAGS(NATOM)
    CHARACTER(len=4) QNRMTYP
    INTEGER   I,WALN
    real(chm_real)  CGMA,TSTA
    real(chm_real)  QRA1,QRB1,QRA2,QRB2,QRQ1,QRQ2,QRK
    LOGICAL   LUSED,QPRINT,QFLEX,QRESET
    LOGICAL   QMASS,WALPP, WTYP
    !
    !
    CALL SELCTA(COMLYN,COMLEN,FLAGS,X,Y,Z,WMAIN,.TRUE.)
    QNRMTYP=GTRMA(COMLYN,COMLEN,'NORM')
    IF (QNRMTYP /= '') THEN
       CALL QPRTN(QNRMTYP,FLAGS,NATOM)
    ENDIF
    QCG=(INDXA(COMLYN,COMLEN,'ON') > 0)
    QCG=(.NOT.(INDXA(COMLYN,COMLEN,'OFF') > 0))
    IF (.NOT.QCG) THEN
       CGMODEL=0
       QCGINV=.FALSE.
       QCGINVF=.FALSE.
       QCG=.FALSE.
       QNOCO=.FALSE.
       QPOLAR1=.FALSE.
       IPOLAR1=-1
       QCGWATER=.FALSE.
    ENDIF
    QRESET=INDXA(COMLYN,COMLEN,'RESE') > 0
    IF (QRESET)  CALL FQRESET(NATOM,ngrp)
    QNOCO=(INDXA(COMLYN,COMLEN,'NOCO') >  0)
    IF (QPRNETA .AND. PRNLEV >= 2) &
         WRITE(OUTU,*) "CGINV will write out Eta matrix"
    CGMODEL=GTRMI(COMLYN,COMLEN,'CGMD',CGMODEL)
    CGEQ=GTRMI(COMLYN,COMLEN,'CGEQ',0)
    QCGINV=INDXA(COMLYN,COMLEN,'CGIN') > 0
    QCGPOL=INDXA(COMLYN,COMLEN,'POLT') > 0
    !      QCGINVF=INDXA(COMLYN,COMLEN,'CGFC') > 0
    QCGWATER=INDXA(COMLYN,COMLEN,'WATE') > 0
    QCGSPC=INDXA(COMLYN,COMLEN,'SPC') > 0
    QFLEX=INDXA(COMLYN,COMLEN,'FLEX') > 0
    QT4P=INDXA(COMLYN,COMLEN,'TIP4') > 0
    IF (QCGWATER)  CALL TYPMOL('WATE',FLAGS)
    IF (QCGSPC) CALL TYPMOL('SPC ',FLAGS)
    IF (QT4P) THEN 
       GQT4P=.TRUE.
       CALL TYPMOL('TIP4',FLAGS)
       if (prnlev >= 2) write(outu,*) " GLOBAL QT4P = ", GQT4P
    ENDIF
    IF (QFLEX) CALL TYPMOL('FLEX',FLAGS)
    QPOLAR1=INDXA(COMLYN, COMLEN, 'QPOL')  >  0
    IPOLAR1=GTRMI(COMLYN,COMLEN,'IPOL',IPOLAR1)
    QPRINT=(INDXA(COMLYN,COMLEN,'PRIN') > 0)
    QPRINT=QPRINT .AND. (PRNLEV >= 2)
    QREF=(INDXA(COMLYN,COMLEN,'QREF') > 0)

    QMASS=INDXA(COMLYN,COMLEN,'QMAS') > 0
    IF (QMASS) THEN
       CGMA=GTRMF(COMLYN,COMLEN,'CGMA',DEFQ)
       TSTA=GTRMF(COMLYN,COMLEN,'TSTA',TCGDEF)

       CALL ASSIGNQMAS(FLAGS,CGMA,TSTA)
    ENDIF
    WTYP=INDXA(COMLYN,COMLEN,'WTYP') > 0
    !     Determine potential type (harmonic or other)
    !     1=Harmonic (preferred); 2=Wall Potential
    IF (WTYP) THEN
       PTYP=GTRMI(COMLYN,COMLEN,'PTYP',1)
       IF (PTYP == 2) THEN
          QRA1=GTRMF(COMLYN,COMLEN,'QRA1',QRA1DEF)
          QRB1=GTRMF(COMLYN,COMLEN,'QRB1',QRB1DEF)
          QRA2=GTRMF(COMLYN,COMLEN,'QRA2',QRA2DEF)
          QRB2=GTRMF(COMLYN,COMLEN,'QRB2',QRB2DEF)
          QRQ1=GTRMF(COMLYN,COMLEN,'QRQ1',QRQ1DEF)
          QRQ2=GTRMF(COMLYN,COMLEN,'QRQ2',QRQ2DEF)
          QRK=GTRMF(COMLYN,COMLEN,'QRK',QRKDEF)
          wallpN=GTRMI(COMLYN,COMLEN,'WALN',1)
       ELSEIF (PTYP == 1) THEN
          QRQ1=GTRMF(COMLYN,COMLEN,'QRQ1',QRQ1DEF)
          QRQ2=GTRMF(COMLYN,COMLEN,'QRQ2',QRQ2DEF)
          QRK=GTRMF(COMLYN,COMLEN,'QRK',QRKDEF)
       ELSE
       ENDIF
       CALL ASSWALLPP(FLAGS,QRA1,QRB1,QRA2,QRB2,QRQ1,QRQ2,QRK, &
            NATOM)
    ENDIF

    CALL XTRANE(COMLYN,COMLEN,'FQCOM')

    IF (QREF) THEN
       CALL FILLCGREF(FLAGS)
    ENDIF

    IF(QPRINT) THEN
       IF (QCGINV) WRITE(OUTU,'(a)') "<FQCOM>: CG INVERSION REQUESTED"
       IF (QCGINVF) WRITE(OUTU,'(a)') "<FQCOM>: CG FORCE REQUESTED"
       WRITE(OUTU,'(a,i6)') "<FQCOM>: CGMODEL SET TO",CGMODEL
       IF (QPOLAR1) WRITE(OUTU,'(2A,I5)') &
            "Single Molecule Polarization is requested for " &
            ,"Molecule #",IPOLAR1
       IF (QNOCO) WRITE(OUTU,'(A)') &
            "Charge only, coordinates being fixed."
       IF (QCGWATER) WRITE(OUTU,'(A)') &
            "Hardness ontribution to Force set to zero"
       DO I=1,NATOM
          WRITE (OUTU,20) I,QPARTBIN(I),QPART(I),QPRTPTR(I)
       ENDDO
       DO I = 1,NATOM
          WRITE(OUTU,*)"ATOM,MOLTYPE=",I,MOLTYPE(I)
       ENDDO
    ENDIF !QPRINT


    !
20  FORMAT(4i9)
    RETURN
  END   SUBROUTINE FQCOM
  !-----------------------------------------------------------------------
  SUBROUTINE FQRESET(NATOM,ngrp)

    INTEGER I,NATOM,ngrp

    QNPART=0
    QCG=.FALSE.
    QPRNETA=.FALSE.
    DO I=1,NATOM
       QPARTBIN(I)=0
       QPART(I)=0
       MOLTYPE(I)=0
       QPARTED(I)=.FALSE.
    ENDDO
    DO I=1,NGRP+1
       QPRTPTR(I)=0
    ENDDO

    RETURN
  END  SUBROUTINE FQRESET
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTN(QNRMTYP,FLAGS,NATOM)
    !  This subroutine is called from FQCOM to define the groups of atoms 
    !  over which charge will be normalized.
    !
    !  QNRMTYP character       type of atom grouping
    !  FLAGS   integer NATOM   
    !
    !  QNPART  integer scalar  total number of partitions     
    !  QPRTPTR integer MAXPART pointers into QPART for end of current partition
    !  QPART   integer NATOM   atom numbers in partition order
    !  QPARTBIN integer NATOM   partition to which atom number belongs
    !      
    use number
    use stream
    !
    INTEGER FLAGS(*),NATOM
    CHARACTER(len=4) QNRMTYP
    INTEGER I,J,PARTCNTR,COUNTATOM
    LOGICAL PARTFOUND
    !
    !
    !  assign new partition numbers to QPARTBIN depending on FLAGS array and
    !  type of partitioning QNRMTYP
    !
    IF     (QNRMTYP == 'BYRE') THEN
       !       CALL QPRTBYRES(FLAGS,QNPART)
       CALL QPRTBYRES(FLAGS,QNPART,QPARTED)
    ELSEIF (QNRMTYP == 'BYAL') THEN
       CALL QPRTBYALL(FLAGS,QNPART)
    ELSEIF (QNRMTYP == 'BYSE') THEN
       CALL QPRTBYSEG(FLAGS,QNPART,QPARTED)
    ELSEIF (QNRMTYP == 'BYGR') THEN
       CALL QPRTBYGRP(FLAGS,QNPART,QPARTED)
    ELSEIF (QNRMTYP == 'BYMO') THEN
       CALL QPRTBYMOL(FLAGS)
    ELSEIF (QNRMTYP == 'NOFQ') THEN
       CALL QPRTNOFQ(FLAGS,QPARTBIN,NATOM)
    ELSEIF (QNRMTYP == 'BYR1') THEN
       CALL QPRTBYR1(FLAGS,QNPART)
    ENDIF

    !  now merge the new elements from the FLAGS array with the old ones 
    !  from the QPARTBIN array
    !
    IF(QNRMTYP /= 'NOFQ') THEN
       CALL PARTMERGE(QPARTBIN,FLAGS,NATOM)
    ENDIF
    !      
    !  counts the number of partitions, compacts the partition bin array, 
    !  and fills the partition occupancy and partition pointer array
    !
    PARTCNTR=1
    QPRTPTR(1)=0
    COUNTATOM=0
    DO I=1,QNPART
       PARTFOUND=.FALSE.
       DO J=1,NATOM
          IF(QPARTBIN(J) == I) THEN
             COUNTATOM=COUNTATOM+1
             QPARTBIN(J)=PARTCNTR
             PARTFOUND=.TRUE.
             QPART(COUNTATOM)=J
          ENDIF
       ENDDO
       IF(PARTFOUND) THEN
          QPRTPTR(PARTCNTR+1)=COUNTATOM
          PARTCNTR=PARTCNTR+1
       ENDIF
    ENDDO
    QNPART=PARTCNTR-1

    IF (QNPART > 1) QMOLC=.TRUE.


    RETURN
  END   SUBROUTINE QPRTN !QNORMAL
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTBYRES(FLAGS,QNPART,QPARTED)
    ! This subroutine is called from QPRTN and sets new partition 
    ! numbers for the FLAGS array by residue.  also sets new QNPART
    use psf
    use stream 

    INTEGER FLAGS(NATOM),QNPART
    INTEGER I,J,PARTCTR,PARTATOMS,RESATOMS
    LOGICAL FOUND,QPARTED(*)
    !
    PARTCTR=1
    DO I=1,NRES
       FOUND=.FALSE.
       PARTATOMS=0
       DO J=(IBASE(I)+1),(IBASE(I+1))

          IF(FLAGS(J) /= 0) THEN
             IF (.not.QPARTED(J)) THEN
                QPARTED(J)=.TRUE.
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ELSE
                if(prnlev > 1)write(outu,49) &
                     '***CHEQ*** Attempting to assign an atom to more than',  &
                     'one normalization unit (RESIDUE); NOT A VALID SCHEME '
                CALL WRNDIE(-1,'<QPRTBYRES>', &
                     'CHEQ: CHECK NORMALIZATION SPECIFICATIONS ')
             ENDIF
          ENDIF
49        format(/,3x,A,/,3x,A)
       ENDDO
       IF (FOUND) THEN
          RESATOMS=IBASE(I+1)-IBASE(I)
          IF(RESATOMS > PARTATOMS) THEN 
             CALL WRNDIE(0,'<FQCOM>','All atoms of residue not selected')
          ENDIF
          PARTCTR=PARTCTR+1
       ENDIF
    ENDDO
    QNPART=QNPART+PARTCTR-1
    !
    RETURN
  END SUBROUTINE QPRTBYRES
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTBYSEG(FLAGS,QNPART,QPARTED)
    ! This subroutine is called from QPRTN and sets new partition 
    ! numbers for the FLAGS array by segment.  also sets new QNPART
    use psf
    use stream 

    INTEGER FLAGS(NATOM),QNPART
    INTEGER I,J,PARTCTR,PARTATOMS,SEGATOMS
    LOGICAL FOUND,QPARTED(*)
    !
    PARTCTR=1
    DO I=1,NSEG
       FOUND=.FALSE.
       PARTATOMS=0
       DO J=(IBASE(NICTOT(I)+1)+1),(IBASE(NICTOT(I+1)+1))
          IF(FLAGS(J) /= 0) THEN
             IF (.not.QPARTED(J)) THEN
                QPARTED(J) = .TRUE.
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ELSE
                if (PRNLEV > 1) write(outu,39) &
                     '***CHEQ*** Attempting to assign an atom to more than', &
                     'one normalization unit (SEGMENT): NOT A VALID SCHEME'
                CALL WRNDIE(-1,'<QPRTBYSEG>', &
                     'CHEQ: CHECK NORMALIZATION SPECIFICATION!')
             ENDIF
          ENDIF
39        format(/,3x,A,/,3x,A)
       ENDDO
       IF (FOUND) THEN
          SEGATOMS=IBASE(NICTOT(I+1)+1)-(IBASE(NICTOT(I)+1))
          IF (SEGATOMS > PARTATOMS) THEN
             CALL WRNDIE(0,'<FQCOM>','All atoms of segment not selected')
          ENDIF
          PARTCTR=PARTCTR+1
       ENDIF
    ENDDO
    QNPART=QNPART+PARTCTR-1
    !
    RETURN
  END  SUBROUTINE QPRTBYSEG
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTBYGRP(FLAGS,QNPART,QPARTED)
    ! This subroutine is called from QPRTN and sets new partition 
    ! numbers for the FLAGS array by segment.  also sets new QNPART

    use psf
    use stream
    INTEGER FLAGS(NATOM),QNPART
    INTEGER I,J,PARTCTR,PARTATOMS,GRPATOMS
    LOGICAL FOUND,QPARTED(*)
    !
    PARTCTR=1
    DO I=1,NGRP
       FOUND=.FALSE.
       PARTATOMS=0
       DO J=(IGPBS(I)+1),(IGPBS(I+1))
          IF(FLAGS(J) /= 0) THEN
             IF (.not.QPARTED(J)) THEN
                QPARTED(J) = .TRUE.
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ELSE
                if(prnlev > 1)write(outu,59) &
                     '***CHEQ*** Attempting to assign an atom to more than',  &
                     'one normalization unit (GROUP); NOT A VALID SCHEME '
                CALL WRNDIE(-1,'<QPRTBYGROUP>', &
                     'CHEQ: CHECK NORMALIZATION SPECIFICATIONS ')
59              format(/,3x,A,/,3x,A)
             ENDIF
          ENDIF
       ENDDO
       IF (FOUND) THEN
          GRPATOMS=IGPBS(I+1)-IGPBS(I)
          IF (GRPATOMS > PARTATOMS) THEN
             CALL WRNDIE(0,'<FQCOM>','All atoms of group not selected')
          ENDIF
          PARTCTR=PARTCTR+1
       ENDIF
    ENDDO
    QNPART=QNPART+PARTCTR-1
    !
    RETURN
  END  SUBROUTINE QPRTBYGRP
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTBYALL(FLAGS,QNPART)
    !  This subroutine is called from QPRTN and sets new partition 
    !  number for trues from FLAGS array.  also sets new QNPART

    use psf
    !
    INTEGER FLAGS(*),QNPART
    INTEGER I
    !
    QNPART=QNPART+1
    DO I=1,NATOM
       IF(FLAGS(I) /= 0) FLAGS(I)=QNPART
    ENDDO
    RETURN
  END SUBROUTINE QPRTBYALL
  !
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTBYMOL(FLAGS)
    !  This subroutine is called from QPRTN and sets new partition
    !  numbers for nonzero values from FLAGS array.  also set new QNPART

    use psf
    use stream
    !      
    INTEGER FLAGS(*)
    INTEGER I,J,PARTCTR,PARTATOMS,MOLATOMS
    LOGICAL FOUND
    !
    PARTCTR=1
    DO I=1,MOLNT
       FOUND=.FALSE.
       PARTATOMS=0
       DO J=MOLPTR(I),(MOLPTR(I+1)-1)
          IF(FLAGS(MOLBL(J)) /= 0) THEN
             IF (.not.QPARTED(MOLBL(J))) THEN
                QPARTED(MOLBL(J))=.TRUE.
                FOUND=.TRUE.
                FLAGS(MOLBL(J))=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ELSE
                if(prnlev > 1)write(outu,29) &
                     '***CHEQ*** Attempting to assign an atom to more than', &
                     'one normalization unit (MOLECULE); NOT A VALID SCHEME '
                CALL WRNDIE(-1,'<QPRTBYMOL>', &
                     'CHEQ: CHECK NORMALIZATION SPECIFICATIONS ')
             ENDIF
          ENDIF
29        format(/,3x,A,/,3x,A)
       ENDDO
       IF (FOUND) THEN
          MOLATOMS=MOLPTR(I+1)-MOLPTR(I)
          IF (MOLATOMS > PARTATOMS) THEN
             CALL WRNDIE(0,'<FQCOM>','All atoms of molecule not selected')
          ENDIF
          PARTCTR=PARTCTR+1
       ENDIF
    ENDDO
    QNPART=QNPART+PARTCTR-1
    !
    RETURN
  END  SUBROUTINE QPRTBYMOL
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTNOFQ(FLAGS,QPARTBIN,NATOM)
    !  This subroutine is called from QPRTN and sets  
    !  

    !
    INTEGER FLAGS(*),QPARTBIN(*),NATOM
    INTEGER I
    !
    DO I=1,NATOM
       IF(FLAGS(I) /= 0) QPARTBIN(I)=0
    ENDDO
    RETURN
  END SUBROUTINE QPRTNOFQ
  !
  !-----------------------------------------------------------------------
  SUBROUTINE QPRTBYR1(FLAGS,QNPART)
    ! This subroutine is called from QPRTN and sets new partition
    ! numbers for the FLAGS array by backbone and sidechain.  also sets new QNPART

    use stream
    use psf

    INTEGER FLAGS(NATOM),QNPART
    INTEGER I,J,PARTCTR,PARTATOMS,RESATOMS

    INTEGER ITHISSEG,NUMRES,FIRRES, LASRES,FIRTYP,LASTYP
    LOGICAL FOUND
    !




    !   loop over NSEG and find which segment to define partitions over 
    !    in this manner


    DO I = 1,NSEG
       J=(IBASE(NICTOT(I)+1)+1)
       IF(FLAGS(J) /= 0) ITHISSEG = I
    ENDDO

    !    Find number of residues in this segment

    NUMRES = NICTOT(ITHISSEG+1) - ( NICTOT(ITHISSEG) ) 
    FIRRES = NICTOT(ITHISSEG)+1
    LASRES = NICTOT(ITHISSEG+1)

    if (prnlev >= 2) then
       write(outu,*) " SEGMENT TO APPLY NORM. METHOD = ", ITHISSEG
       write(outu,*) " Number of residues = ", NUMRES
       write(outu,*) " First , Last Residue Number =", FIRRES, LASRES
       write(outu,*) " NRES, QNPART  = ", NRES, QNPART
    endif

    IF (NUMRES >= 3) THEN

       PARTCTR=1
       FIRTYP=1

       IF (FIRTYP == 1) THEN   ! ACE

          FOUND=.FALSE.
          PARTATOMS=0
          I = FIRRES
          DO J = (IBASE(I)+1), (IBASE(I)+6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR=PARTCTR+1
          ENDIF

          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+7), (IBASE(I)+12)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO


          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF


          FOUND=.FALSE.
          PARTATOMS=0

          DO J = (IBASE(I)+13), (IBASE(I+1)) !  (IBASE(I+1)-2)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

       ELSEIF (FIRTYP == 2) THEN ! NTER


          FOUND=.FALSE.
          PARTATOMS=0
          I = FIRRES
          DO J = (IBASE(I)+1), (IBASE(I)+8)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR=PARTCTR+1
          ENDIF

          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+9), (IBASE(I+1))
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF


       ELSE 
       ENDIF  !  END CONDITIONAL FOR FIRTYP FOR FIRST RESIDUE

       DO I=FIRRES+1,LASRES-1
          FOUND=.FALSE.
          PARTATOMS=0
          DO J=(IBASE(I)+1),(IBASE(I)+6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO

          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF
          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+7), (IBASE(I+1))
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF
       ENDDO



       !    for first residue of this segment
       !     this is assumed to be patched with some patch residue
       !    the choices for the patch are hacked via hard-coding
       !    first patches accounted for are:   FIRTYP=1(ACE)   FIRTYP=2(NTER)
       !    last  patches accounted for are:   FIRTYP=1(CT3)   FIRTYP=2(CTER)
       !     these are taken to be common combinations of patch residues
       !    can be generalized later



       IF (FIRTYP == 1) THEN

          I = LASRES      ! CT3


          FOUND=.FALSE.
          PARTATOMS=0
          DO J= (IBASE(I)+1), (IBASE(I)+6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO

          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+7), (IBASE(I+1)-6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I+1)-5), (IBASE(I+1))
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

       ELSE  IF (FIRTYP == 2) THEN
       ELSE
       ENDIF


       QNPART=QNPART+PARTCTR-1


    ELSEIF (NUMRES == 1) THEN

       !      
       PARTCTR=1
       FIRTYP = 1 
       IF (FIRTYP == 1) THEN
          FOUND=.FALSE.
          PARTATOMS=0
          I = FIRRES
          DO J = (IBASE(I)+1), (IBASE(I)+6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF


          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+7), (IBASE(I)+12)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF



          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+13), (IBASE(I+1)-6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF


          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I+1)-5), (IBASE(I+1))
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

       ELSE
       ENDIF  ! CONDITOINAL FOR FIRTYP

       QNPART = QNPART + PARTCTR - 1



    ELSEIF (NUMRES == 2) THEN

       PARTCTR=1
       FIRTYP=1

       IF (FIRTYP == 1) THEN

          I = FIRRES
          FOUND=.FALSE.
          PARTATOMS=0
          I = FIRRES
          DO J = (IBASE(I)+1), (IBASE(I)+6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF


          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+7), (IBASE(I)+12)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF



          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+13), (IBASE(I+1))
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF



          I = LASRES

          FOUND=.FALSE.
          PARTATOMS=0
          DO J= (IBASE(I)+1), (IBASE(I)+6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO

          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I)+7), (IBASE(I+1)-6)
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

          FOUND=.FALSE.
          PARTATOMS=0
          DO J = (IBASE(I+1)-5), (IBASE(I+1))
             IF(FLAGS(J) /= 0) THEN
                FOUND=.TRUE.
                FLAGS(J)=QNPART+PARTCTR
                PARTATOMS=PARTATOMS+1
             ENDIF
          ENDDO
          IF (FOUND) THEN
             PARTCTR = PARTCTR + 1
          ENDIF

       ELSE
       ENDIF ! CONDITIONAL on FIRTYP

       QNPART = QNPART + PARTCTR - 1

    ELSE
    ENDIF  ! CONDITIONAL on NUMRES

    RETURN
  END     SUBROUTINE QPRTBYR1

  SUBROUTINE PARTMERGE(QPRTBIN,NEWPART,NATOM)
    ! This subroutine is called from QPRTN and merges the new parts of
    ! FLAGS (i.e. nonzero values) into the QPRTBIN array
    !

    !
    INTEGER NEWPART(*),QPRTBIN(*),NATOM
    INTEGER I
    !
    DO I=1,NATOM
       IF(NEWPART(I) /= 0) QPRTBIN(I)=NEWPART(I)
    ENDDO
    RETURN
  END  SUBROUTINE PARTMERGE
  !
  !-----------------------------------------------------------------------
  ! zz12-02-95
  SUBROUTINE MOLSRT(NATOM,IB,JB,NBOND,QCGRP,NGRP,IGPBS)
    !
    ! This routine sorts out molecules defined by bonds. Atoms with same
    ! MOLT are linked to each other through chemical bonds
    !
    !  Zhou 12-02-95

    use stream
    INTEGER NATOM,IB(:),JB(:),NBOND,IGPBS(:),NGRP
    LOGICAL QCGRP
    !  Local variable
    INTEGER BLIST(13,NATOM)
    INTEGER I,J,K,M,IS,IQ

    IF (QCGRP) THEN
       ! use groups as molecules
       M=1
       DO I=1,NGRP
          MOLPTR(I)=M
          IS=IGPBS(I)+1
          IQ=IGPBS(I+1)
          DO J=IS,IQ
             MOLT(J)=I
             MOLBL(M)=J
             M=M+1
          ENDDO
       ENDDO
       K=NGRP
       ! This is to make sure the last molecule has the right # of atoms
       MOLPTR(K+1)=M
       IF(M /= (NATOM+1)) CALL WRNDIE(-3,'MOLSRT>', &
            'Some atoms have not assigned a mol. #.')
       ! test printout
       !         WRITE(OUTU,'(A)')"I,MOLT,MOLPTR,MOLBL:"
       !         DO I=1,NATOM
       !           WRITE(OUTU,'(4(2x,I9))')I,MOLT(I),MOLPTR(I),MOLBL(I)
       !         ENDDO
       MOLNT = K
       RETURN
    ENDIF

    ! BLIST(1,J) = 1 + count of higher-numbered atoms bonded to atom J
    ! BLIST(2:,J) = list of those atoms
    BLIST(1,:) = 1
    DO I=1,NBOND
       IF (IB(I) > JB(I)) THEN
          J=JB(I)
          K=IB(I)
       ELSE
          J=IB(I)
          K=JB(I)
       ENDIF
       IF (J < 1 .OR. J > NATOM .OR. K < 1 .OR. K > NATOM) &
            CYCLE
       BLIST(1,J)=BLIST(1,J)+1
       IF (BLIST(1,J) > 13) THEN
          WRITE(OUTU,900)J,(BLIST(M,J),M=2,13),K
900       FORMAT(/,1X,'<MOLSRT>: More than 12 atoms bonded to atom ', &
               I8,':',/,10X,7(2X,I8),/)
          CALL WRNDIE(-3,'MOLSRT>', &
               'More than 12 bonds found for an atom')
       ENDIF
       BLIST(BLIST(1,J),J)=K
    ENDDO
    ! test printout
    !      DO I=1,NATOM
    !        WRITE(OUTU,'(7(2X,I5))')I,(BLIST(J,I),J=2,BLIST(1,I))
    !      ENDDO

    ! K = count of distinct molecules
    ! MOLT(I) = which molecule contains atom I
    MOLT = 0
    K = 0
    DO I = 1, NATOM
       IF (MOLT(I) == 0) THEN
          ! new molecule
          K = K + 1
          MOLT(I) = K
       ENDIF
       DO J = 2, BLIST(1,I)
          MOLT(BLIST(J,I)) = MOLT(I)
       ENDDO
    ENDDO

    ! Rearrange atom so all atoms with the same MOLT will be in consecutive
    ! block in MOLBL. MOLPTR points to the first atom of a molecule in MOLBL
    MOLPTR = 0
    MOLBL = 0
    M=1
    DO I=1,K
       MOLPTR(I)=M
       DO J=1,NATOM
          IF (MOLT(J) == I) THEN
             MOLBL(M)=J
             M=M+1
          ENDIF
       ENDDO
    ENDDO
    ! This is to make sure the last molecule has the right # of atoms
    MOLPTR(K+1)=M
    IF(M /= (NATOM+1)) CALL WRNDIE(-3,'MOLSRT>', &
         'Some atoms have not assigned a mol. #.')
    ! test printout
    !      WRITE(OUTU,'(/,A,I8,/,A)')'MOLNT = ',K,'I,MOLT,MOLPTR,MOLBL'
    !      WRITE(OUTU,'(4(2X,I9))')(I,MOLT(I),MOLPTR(I),MOLBL(I),I=1,NATOM)
    !
    MOLNT = K
    RETURN
  END SUBROUTINE MOLSRT
  !-----------------------------------------------------------------------
  SUBROUTINE TYPMOL(MODE,FLAGS)

    use stream
    use param
    use psf
    use comand
    use coord
    use memory
    use select

    CHARACTER(len=4),intent(in) :: MODE
    INTEGER I, MARK,J,J1,JMD,JMT,K
    INTEGER IWN(5),FLAGS(natom)
    integer,allocatable,dimension(:) :: IW

    if (prnlev >= 2) write(outu,*) 'TYPMOL: Setting selection to type ',MODE
    DO I=1,NATOM
       IF (FLAGS(I) > 0) THEN 
          MOLTYPE(I)=0
       ENDIF
    ENDDO
    IF (MODE == 'WATE') THEN
       MARK=-1
    ELSEIF (MODE == 'SPC ') THEN
       MARK=1
    ELSEIF (MODE == 'TIP4') THEN
       MARK=3
    ELSE  
       RETURN
    END IF
    ! Just in case
    call chmalloc('cheqmodule.src','TYPMOL','IW',NATOM,intg=IW)
    CALL SELCTA(COMLYN,COMLEN,IW,X,Y,Z,WMAIN,.FALSE.)

    IF (MODE == 'SPC') THEN

       DO I=1,NATOM
          IF(FLAGS(I) > 0) THEN
             IF ((IW(I) == 1).AND.(MOLTYPE(I).EQ.0)) THEN
                JMT=MOLT(I)
                IF ((MOLPTR(JMT+1)-MOLPTR(JMT)) == 3) THEN  !must be water!
                   K=1
                   DO JMD=MOLPTR(JMT),MOLPTR(JMT+1)-1
                      J=MOLBL(JMD)
                      IWN(K)=J
                      K=K+1
                   ENDDO
                   IF (ITC(IAC(IWN(1))) == ITC(IAC(IWN(2)))) THEN
                      MOLTYPE(IWN(1))=2*MARK
                      MOLTYPE(IWN(2))=2*MARK
                      MOLTYPE(IWN(3))=MARK
                   ELSE IF (ITC(IAC(IWN(1))) == ITC(IAC(IWN(3)))) THEN
                      MOLTYPE(IWN(1))=2*MARK
                      MOLTYPE(IWN(2))=MARK
                      MOLTYPE(IWN(3))=2*MARK
                   ELSE IF (ITC(IAC(IWN(2))) == ITC(IAC(IWN(3)))) THEN
                      MOLTYPE(IWN(1))=MARK
                      MOLTYPE(IWN(2))=2*MARK
                      MOLTYPE(IWN(3))=2*MARK
                   ELSE
                      CALL WRNDIE(-3,'<MOLTYP>','It is not Water.')
                   END IF
                END IF
             END IF
          END IF
       ENDDO

    ELSEIF (MODE == 'TIP4') THEN

       DO I=1,NATOM
          IF(FLAGS(I) > 0) THEN
             IF ( (IW(I) == 1 ).AND.(MOLTYPE(I).EQ.0)) THEN
                JMT=MOLT(I)
                IF ((MOLPTR(JMT+1)-MOLPTR(JMT)) == 4) THEN  !must be 4-point water!
                   K=1
                   DO JMD=MOLPTR(JMT),MOLPTR(JMT+1)-1
                      J=MOLBL(JMD)
                      IWN(K)=J
                      K=K+1
                   ENDDO
                   !       assume tip4p with following order of atoms:  OH2, OM, H1, H2
                   !       OM is the lone pair which carries the charge and undergoes
                   !       fluctuating charge dynamics
                   !       hardcoded for now
                   !       S. Patel: (08/28/02)

                   MOLTYPE(IWN(1))=9
                   MOLTYPE(IWN(2))=3
                   MOLTYPE(IWN(3))=4
                   MOLTYPE(IWN(4))=4

                   !               IF (ITC(IAC(IWN(1))) == ITC(IAC(IWN(2)))) THEN
                   !                 MOLTYPE(IWN(1))=2*MARK
                   !                 MOLTYPE(IWN(2))=2*MARK
                   !                 MOLTYPE(IWN(3))=MARK
                   !               ELSE IF (ITC(IAC(IWN(1))) == ITC(IAC(IWN(3)))) THEN
                   !                 MOLTYPE(IWN(1))=2*MARK
                   !                 MOLTYPE(IWN(2))=MARK
                   !                 MOLTYPE(IWN(3))=2*MARK
                   !               ELSE IF (ITC(IAC(IWN(2))) == ITC(IAC(IWN(3)))) THEN
                   !                 MOLTYPE(IWN(1))=MARK
                   !                 MOLTYPE(IWN(2))=2*MARK
                   !                 MOLTYPE(IWN(3))=2*MARK
                   !               ELSE
                   !                 CALLWRNDIE(-3,'<MOLTYP>','It is not Water.')
                   !               END IF
                END IF
             END IF
          END IF
       ENDDO

    ELSE
    ENDIF

    call chmdealloc('cheqmodule.src','TYPMOL','IW',NATOM,intg=IW)
    RETURN
  END SUBROUTINE TYPMOL
  !-----------------------------------------------------------------------
  SUBROUTINE FILLCGREF(FLAGS)

    use coord
    use psf

    INTEGER FLAGS(*)
    INTEGER I

    DO I=1,NATOM
       IF(FLAGS(I) /= 0) CGREF(I)=WMAIN(I)
    ENDDO

    RETURN
  END SUBROUTINE FILLCGREF
  !---------------------------------------------------------
  SUBROUTINE ASSIGNQMAS(FLAGS,QMA,TSTA)

    use coord
    use psf
    use consta

    INTEGER FLAGS(*)
    INTEGER I
    real(chm_real)  QMA,TSTA

    DO I=1,NATOM
       IF(FLAGS(I) /= 0) THEN 
          PMASSQ(I)=QMA/(TIMFAC*TIMFAC)
       ENDIF
    ENDDO
    RETURN
  END  SUBROUTINE ASSIGNQMAS
  !-----------------------------------------------------------------------
  SUBROUTINE ASSWALLPP(FLAGS,QRA1,QRB1,QRA2,QRB2,QRQ1,QRQ2,QRK, &
       NATOM)

    !
    INTEGER FLAGS(*)
    INTEGER I,NATOM
    real(chm_real) QRA1,QRB1,QRA2,QRB2,QRQ1,QRQ2,QRK
    !
    !
    DO I=1,NATOM
       IF(FLAGS(I) /= 0) THEN
          pwallpa1(I)=QRA1
          pwallpb1(I)=QRB1
          pwallpa2(I)=QRA2
          pwallpb2(I)=QRB2
          pwallpQ1(I)=QRQ1
          pwallpQ2(I)=QRQ2
          pwallpK(I)=QRK
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE ASSWALLPP



  ! --

  SUBROUTINE CHEQRESETNORM

    !     reset the QPARTED array used in checking for assignmetn of atoms
    !     to more than one normalization unit

    use number
    use coord
    use stream
    use psf

    INTEGER I

    DO I = 1, NATOM
       QPARTED(I)=.FALSE.
    ENDDO
    RETURN
  END SUBROUTINE CHEQRESETNORM

  !
  !CC The unused dummy variables are here in the following two routines
  !CC for later development and consistency with their original routines.
  !-----------------------------------------------------------------------
  SUBROUTINE GETVR1Q(VARB,CG)
    !     Put the coordinates and crystal variables into the variable array.
    !
    !
    REAL(CHM_REAL)      VARB(*), CG(*)
    !
    INTEGER     I, IP, J
    !
    IP = 0

    DO I=1,QNPART
       DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
          IP=IP+1
          VARB(IP)=CG(QPART(J))
       ENDDO
    ENDDO

    RETURN
  END subroutine getvr1q

  !-----------------------------------------------------------------------
  SUBROUTINE PUTVR1Q(VARB,CG,CG_FIX,CGT)
    !     Fill the coordinate and crystal arrays with the variables.


    REAL(CHM_REAL)      VARB(*),CG(*)
    REAL(CHM_REAL)      CG_FIX(*),CGT(*)


    INTEGER     I, IP, J
    REAL(CHM_REAL)  QDIF

    IP = 0

    DO I=1,QNPART
       CGT(I)=0.0
       DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
          IP=IP+1
          CGT(I)=CGT(I)+VARB(IP)
          CG(QPART(J))=VARB(IP)
       ENDDO
       !        QDIF=(CG_FIX(I)-CGT(I))/((QPRTPTR(I)+1)-QPRTPTR(I+1))
       !        DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
       !           CG(QPART(J))=CG(QPART(J))-QDIF
       !        ENDDO
    ENDDO

    RETURN
  END subroutine putvr1q

  !-----------------------------------------------------------------------
  SUBROUTINE MOLCGT(NATOM,CG,CG_FIX)
    !  This routine computes the total charge on each 
    !  partition over which charge is nomalized.
    REAL(CHM_REAL) CG(*),CG_FIX(*)
    INTEGER NATOM
    INTEGER I, I1,J,J1
    DO I=1,QNPART
       CG_FIX(I)=0.0
       DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
          CG_FIX(I)=CG_FIX(I)+CG(QPART(J))
       ENDDO
       !      write(6,'(A,I5,F15.4)')"CGFIX",I,CG_FIX(I)
    ENDDO

    RETURN

  end subroutine molcgt


  !-----------------------------------------------------------------------
  !      routines from fqener
  !-----------------------------------------------------------------------
  SUBROUTINE QCGSET(NATOM,X,Y,Z,DX,DY,DZ,CG,INBLO,IBLO14,INB14,JNB, &
       IAC,LELEC, &
#if KEY_WCA==1
       LLSOFT,RCUT,WCA, &     
#endif
       OUTU)

    !-----------------------------------------------------------------------
    !  This routine sets the charges, forces and second derivatives at the
    !  beginning of the energy call.

    use nb_module
    use number
    use fast
#if KEY_PARALLEL==1
    use parallel
#endif 
    INTEGER NATOM,IAC(*),OUTU,JJ,thisisjunk
    INTEGER INBLO(*),IBLO14(*),INB14(*),JNB(*)
    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CG(*)
    LOGICAL LELEC
    !
    !
    !      real(chm_real),allocatable,dimension(:) ::

    INTEGER I,NDA
    INTEGER ETASIZE
    INTEGER inter,Rpoint,delR,temp
    real(chm_real),allocatable,dimension(:,:) :: datmp

#if KEY_WCA==1
    REAL(CHM_REAL) WCA(*)
    REAL(CHM_REAL) RCUT
    LOGICAL LLSOFT
#endif 

    IF (LELEC) THEN
       IF (QCGPOL) THEN
          write(OUTU,*)" CALING POLT"

          CALL FQPOLTEN(NATOM,X,Y,Z,DX,DY,DZ,CG,ECH,EHA,QMOLC,MOLT, &
               DCH,QCGINVF,JNB, &
               INBLO,INB14,IBLO14,IAC,QPARTBIN,MOLTYPE,QPRNETA, &
               CCNBA,CCNBB, &
               CCNBC,CCNBD,IACNB, &
               NITCC2,LOWTP,OUTU)
          deallocate(datmp)
       ENDIF
       IF (QCGINV) THEN
          ETASIZE=NATOM*NATOM
          IF (QCGINVF) THEN
             NDA=NATOM
          ELSE
             NDA=1
          ENDIF
          allocate(datmp(nda,nda),stat=alloc_err)

          CALL CGINV(NATOM,X,Y,Z,DX,DY,DZ,CG,ECH,EHA,QMOLC,MOLT, &
               DATMP,DCH,QCGINVF,JNB, &
               INBLO,INB14,IBLO14,IAC,QPARTBIN,MOLTYPE,QPRNETA, &
               CCNBA,CCNBB,CCNBC, &
               CCNBD,IACNB, &
               NITCC2,LOWTP, &
#if KEY_WCA==1
               LLSOFT,RCUT,WCA, & 
#endif
               OUTU)

          DO I=1,NATOM
             CG(I)=DCH(I)
             DCH(I)=ECH(I)
             DDCH(I)=ZERO
          ENDDO
          deallocate(datmp)
          QCGINV=.FALSE.
       ELSE
          DO I=1,NATOM
             DCH(I)=ECH(I)
             DDCH(I)=ZERO
          ENDDO
       ENDIF
    ELSE
       DO I=1,NATOM
          DCH(I)=ZERO
          DDCH(I)=ZERO
       ENDDO
    ENDIF


    RETURN
  END  SUBROUTINE QCGSET

  ! SUBROUTINE QNORM(DCH,QT4P)
  !   !-----------------------------------------------------------------------
  !   !  This subroutine keeps the charge for the given partitions constant
  !   !  over a partition by setting the net derivative with respect to
  !   !  charge to zero.

  !   use number

  !   real(chm_real)  DCH(*)

  !   INTEGER I,J,NAT(10),frstatom,IFLAG
  !   real(chm_real) PARTDCH,TMASS,MASINV
  !   LOGICAL MINNORM,QT4P


  !   IFLAG=2
  !   IF (IFLAG == 1) THEN
  !      NAT(1)=4
  !      NAT(2)=2
  !      NAT(3)=4
  !      NAT(4)=2
  !      NAT(5)=4
  !      NAT(6)=4
  !      NAT(7)=2
  !      NAT(8)=4
  !      NAT(9)=4
  !      NAT(10)=3

  !      frstatom=0
  !      DO I=1,10
  !         PARTDCH=ZERO
  !         DO J=frstatom+1,frstatom+NAT(I)
  !            PARTDCH=PARTDCH + DCH(J)
  !         ENDDO
  !         PARTDCH=PARTDCH/(NAT(I))
  !         DO J=frstatom+1,frstatom+NAT(I)
  !            DCH(J)= DCH(J)-PARTDCH
  !         ENDDO
  !         frstatom= J - 1
  !      ENDDO

  !   ELSEIF(IFLAG == 2) THEN


  !      IF (MINNORM) THEN



  !         IF (QT4P ) THEN

  !            DO I=1,QNPART
  !               PARTDCH=ZERO

  !               IF ( ((QPRTPTR(I+1)-QPRTPTR(I)) == 4) ) THEN ! .AND.
  !                  !     2      (MOLTYPE(QPRTPTR(I)+1) == 9) ) THEN

  !                  DO J=QPRTPTR(I)+2,QPRTPTR(I+1)
  !                     PARTDCH=PARTDCH+DCH(QPART(J))
  !                  ENDDO !J
  !                  PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I)-1)
  !                  !      Write(6,*)' Correction for partition ', i, ' = ', partdch
  !                  DO J=QPRTPTR(I)+2,QPRTPTR(I+1)
  !                     DCH(QPART(J))=DCH(QPART(J))-PARTDCH
  !                  ENDDO !J

  !               ELSE

  !                  DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                     PARTDCH=PARTDCH+DCH(QPART(J))
  !                  ENDDO !J
  !                  PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I))
  !                  !      Write(6,*)' Correction for partition ', i, ' = ', partdch
  !                  DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                     DCH(QPART(J))=DCH(QPART(J))-PARTDCH
  !                  ENDDO !J

  !               ENDIF

  !            ENDDO !I



  !         ELSE

  !            DO I=1,QNPART
  !               PARTDCH=ZERO
  !               DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                  PARTDCH=PARTDCH+DCH(QPART(J))
  !               ENDDO !J
  !               PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I))
  !               !      Write(6,*)' Correction for partition ', i, ' = ', partdch
  !               DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                  DCH(QPART(J))=DCH(QPART(J))-PARTDCH
  !               ENDDO !J
  !            ENDDO !I


  !         ENDIF  ! CONDITIONAL ON QT4P

  !      ELSE



  !         IF (QT4P) THEN

  !            DO I=1,QNPART
  !               TMASS=ZERO
  !               PARTDCH=ZERO

  !               IF ( ((QPRTPTR(I+1)-QPRTPTR(I)) == 4)) THEN  !   .AND.
  !                  !     2      (MOLTYPE(QPRTPTR(I)+1) == 9) ) THEN

  !                  DO J=QPRTPTR(I)+2,QPRTPTR(I+1) ! Starte with OM site on TIP4P hardwired topology
  !                     MASINV=ONE/PMASSQ(QPART(J))
  !                     PARTDCH=PARTDCH+DCH(QPART(J))*MASINV
  !                     !           PARTDCH=PARTDCH+DCH(QPART(J))
  !                     TMASS = TMASS + MASINV
  !                  ENDDO !J
  !                  PARTDCH=PARTDCH/TMASS
  !                  DO J=QPRTPTR(I)+2,QPRTPTR(I+1) ! Start with OM site on TIP4P hardwired topology
  !                     DCH(QPART(J))=DCH(QPART(J))-PARTDCH
  !                  ENDDO !J

  !               ELSE

  !                  DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                     MASINV=ONE/PMASSQ(QPART(J))
  !                     PARTDCH=PARTDCH+DCH(QPART(J))*MASINV
  !                     TMASS = TMASS + MASINV
  !                  ENDDO !J
  !                  PARTDCH=PARTDCH/TMASS
  !                  DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                     DCH(QPART(J))=DCH(QPART(J))-PARTDCH
  !                  ENDDO !J

  !               ENDIF

  !            ENDDO !I

  !         ELSE

  !            DO I=1,QNPART
  !               TMASS=ZERO
  !               PARTDCH=ZERO
  !               DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                  MASINV=ONE/PMASSQ(QPART(J))
  !                  PARTDCH=PARTDCH+DCH(QPART(J))*MASINV
  !                  !           PARTDCH=PARTDCH+DCH(QPART(J))
  !                  TMASS = TMASS + MASINV
  !               ENDDO !J
  !               !        PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I))
  !               PARTDCH=PARTDCH/TMASS
  !               DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
  !                  !          DCH(QPART(J))=DCH(QPART(J))-PMASSQ(QPART(J))*PARTDCH
  !                  DCH(QPART(J))=DCH(QPART(J))-PARTDCH
  !               ENDDO !J
  !            ENDDO !I

  !            !             ENDIF

  !         ENDIF  !  CONDITONIAL on Q4TP



  !      ENDIF  ! MINNORM CONDITIONAL

  !   ELSEIF(IFLAG == 3) THEN

  !      !

  !   ENDIF

  !   RETURN
  ! END   SUBROUTINE QNORM

  SUBROUTINE QNORM(DCH, local_QT4P)
    !-----------------------------------------------------------------------
    !  This subroutine keeps the charge for the given partitions constant
    !  over a partition by setting the net derivative with respect to
    !  charge to zero.

    use chm_kinds, only: chm_real
    use number, only: zero, one

    implicit none

    real(chm_real), intent(inout) ::  DCH(*)
    LOGICAL, intent(in) :: local_QT4P

    INTEGER I,J,NAT(10),frstatom,IFLAG
    real(chm_real) PARTDCH,TMASS,MASINV

    IF (MINNORM) THEN
       IF (local_QT4P) THEN

          DO I=1,QNPART
             PARTDCH=ZERO

             IF ( ((QPRTPTR(I+1)-QPRTPTR(I)) == 4) ) THEN

                DO J=QPRTPTR(I)+2,QPRTPTR(I+1)
                   PARTDCH=PARTDCH+DCH(QPART(J))
                ENDDO !J
                PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I)-1)
                DO J=QPRTPTR(I)+2,QPRTPTR(I+1)
                   DCH(QPART(J))=DCH(QPART(J))-PARTDCH
                ENDDO !J

             ELSE

                DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                   PARTDCH=PARTDCH+DCH(QPART(J))
                ENDDO !J
                PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I))
                DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                   DCH(QPART(J))=DCH(QPART(J))-PARTDCH
                ENDDO !J

             ENDIF

          ENDDO !I

       ELSE ! .not. local_qt4p

          DO I=1,QNPART
             PARTDCH=ZERO
             DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                PARTDCH=PARTDCH+DCH(QPART(J))
             ENDDO !J
             PARTDCH=PARTDCH/(QPRTPTR(I+1)-QPRTPTR(I))
             DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                DCH(QPART(J))=DCH(QPART(J))-PARTDCH
             ENDDO !J
          ENDDO !I

       ENDIF  ! CONDITIONAL ON QT4P
    ELSE ! .not. minnorm
       IF (local_QT4P) THEN

          DO I=1,QNPART
             TMASS=ZERO
             PARTDCH=ZERO

             IF ( ((QPRTPTR(I+1)-QPRTPTR(I)) == 4)) THEN

                DO J=QPRTPTR(I)+2,QPRTPTR(I+1) ! Starte with OM site on TIP4P hardwired topology
                   MASINV=ONE/PMASSQ(QPART(J))
                   PARTDCH=PARTDCH+DCH(QPART(J))*MASINV
                   TMASS = TMASS + MASINV
                ENDDO !J
                PARTDCH=PARTDCH/TMASS
                DO J=QPRTPTR(I)+2,QPRTPTR(I+1) ! Start with OM site on TIP4P hardwired topology
                   DCH(QPART(J))=DCH(QPART(J))-PARTDCH
                ENDDO !J

             ELSE

                DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                   MASINV=ONE/PMASSQ(QPART(J))
                   PARTDCH=PARTDCH+DCH(QPART(J))*MASINV
                   TMASS = TMASS + MASINV
                ENDDO !J
                PARTDCH=PARTDCH/TMASS
                DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                   DCH(QPART(J))=DCH(QPART(J))-PARTDCH
                ENDDO !J

             ENDIF

          ENDDO !I

       ELSE ! .not. local_qt4p

          DO I=1,QNPART
             TMASS=ZERO
             PARTDCH=ZERO
             DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                MASINV=ONE/PMASSQ(QPART(J))
                PARTDCH=PARTDCH+DCH(QPART(J))*MASINV
                TMASS = TMASS + MASINV
             ENDDO !J
             PARTDCH=PARTDCH/TMASS
             DO J=QPRTPTR(I)+1,QPRTPTR(I+1)
                DCH(QPART(J))=DCH(QPART(J))-PARTDCH
             ENDDO !J
          ENDDO !I

       ENDIF  !  CONDITONIAL on Q4TP
    ENDIF  ! MINNORM CONDITIONAL
    RETURN
  END   SUBROUTINE QNORM

  SUBROUTINE QBENER(NATOM,CG,X,Y,Z,DX,DY,DZ,IBLO14,INB14,IAC, &
       QSECD,LEWALD,LELEC)
    !-----------------------------------------------------------------------
    !  This subroutine calculates the electrostatic energy contribution of
    !  the pairs on the non-bonded exclusion list.
    !
    use number
    use parallel
    use energym
    use param
    use stream
    use consta
    use machutil, only: die
    use param_store, only: set_param

    implicit none

    INTEGER NATOM,IAC(*),IBLO14(*),INB14(*)
    real(chm_real) CG(*)
    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    LOGICAL QSECD,LEWALD,LELEC
    !

    INTEGER I,J,JJ,NBINDX,JCTR,JCOUNT,thisisjunk,KK,KJ
    INTEGER IHIJ,IMASS,JMASS,ISTEP,IFLAG
    real(chm_real) HIJ,HIJ1,CGT,FI,CGF,ECSUM,ECSUM1,ECSUM2,ENPB,S
    real(chm_real) CRXI,CRYI,CRZI,DXIJ,DYIJ,DZIJ,FXI,FYI,FZI
    real(chm_real) FJ,FXJ,FYJ,FZJ,CGTJ,sum1 

    real(chm_real) term1,term2,term3,swchfunc,myeterm
    real(chm_real) term4,myforce,term2f,BRACK,QGP,ETILD
    real(chm_real) B(12)
    !

433 FORMAT(' QBENER: ',A)
435 FORMAT(20I5)


#if KEY_PARALLEL==1

    IF (MYNOD == 0) THEN
#endif 

       sum1 = 0.0

       ECSUM = ZERO
       ECSUM1=ZERO
       ECSUM2 = ZERO
       !     main looping structure
       IF(CGMODEL == 0) THEN
          !     No CG-XYZ mixing
          DO I=1,NATOM
             ECSUM=ECSUM+ECH(I)*CG(I)
             HIJ=EHA(I)+EHA(I)
             ECSUM1=ECSUM1+CG(I)*HIJ*CG(I)
             DCH(I)=DCH(I)+HIJ*CG(I)
          ENDDO
          DO I=1,NATOM-1
             IF (I == 1) THEN
                NBINDX=1
             ELSE
                NBINDX=IBLO14(I-1)+1
             ENDIF
             JCTR=IBLO14(I)
             DO JJ=NBINDX,JCTR
                J=IABS(INB14(JJ))
                IF (J > 0) THEN
                   S=(X(I)-X(J))*(X(I)-X(J))+(Y(I)-Y(J))*(Y(I)-Y(J)) &
                        +(Z(I)-Z(J))*(Z(I)-Z(J))
                   HIJ=CCELEC*HIJ/SQRT((1.0+HIJ*HIJ*S))
                   DCH(I)=DCH(I)+HIJ*CG(J)
                   DCH(J)=DCH(J)+HIJ*CG(I)
                   IF(QREF) THEN
                      DCH(I)=DCH(I)+HIJ*CGREF(J)
                      DCH(J)=DCH(J)+HIJ*CGREF(I)
                   ENDIF
                ENDIF
             ENDDO            !JJ
          ENDDO               ! I
       ELSE IF (CGMODEL == 1)  THEN
          NBINDX=1
          !     electrochemical potential term and diagonal hardness term,
          !     hardness term
          DO I=1,NATOM
             IF (QPARTBIN(I) /= 0) THEN
                IF (MOLTYPE(I) <= 0) THEN
                   ECSUM=ECSUM+ECH(I)*CG(I)
                   HIJ=EHA(I)+EHA(I)
                   ECSUM1=ECSUM1+CG(I)*HIJ*CG(I)*CCELEC
                   DCH(I)=DCH(I)+HIJ*CG(I)*CCELEC

                   IF(QREF) THEN
                      DCH(I)=DCH(I)+HIJ*CGREF(I)
                   ENDIF
                ELSEIF (MOLTYPE(I) > 0) THEN ! for water molecules
                   IF (MOLTYPE(I) == 1) THEN
                      HIJ=367.0
                   ELSEIF (MOLTYPE(I) == 2) THEN
                      HIJ=392.2
                   ELSEIF (MOLTYPE(I) == 3) THEN
                      HIJ = 371.6 !  TIP4P OM SITE
                   ELSEIF (MOLTYPE(I) == 4) THEN
                      HIJ = 353.0 ! TIP4P HYDROGEN SITE
                   ELSEIF (MOLTYPE(I) == 9) THEN
                      HIJ = 0.0 !   TIP4P oxygen site (no charge)
                   ELSE
                   ENDIF
                   ECSUM=ECSUM+ECH(I)*CG(I)
                   ECSUM1=ECSUM1+CG(I)*HIJ*CG(I)
                   DCH(I)=DCH(I)+HIJ*CG(I)
                   IF(QREF) THEN
                      DCH(I)=DCH(I)+HIJ*CGREF(I)
                   ENDIF
                ENDIF

                !___________________________________________________________________________
                !     add in wall potential energy

                IF (PTYP == 1) THEN
                   !     Harmonic potential
                   IF (CG(I) < pwallpQ2(I)) THEN                        
                      myeterm = pwallpK(I)*(CG(I)-pwallpQ2(I))* &
                           (CG(I)-pwallpQ2(I))
                      myforce=1.0*2.0*pwallpK(I)*(CG(I)-pwallpQ2(I))

                      ECSUM = ECSUM + myeterm
                      DCH(I) = DCH(I) + myforce


                   ELSE IF ((CG(I) >= pwallpQ2(I)) .and. &
                        (CG(I) <= pwallpQ1(I))) THEN

                      myeterm = 0.0
                      myforce = 0.0

                      ECSUM = ECSUM + myeterm
                      DCH(I) = DCH(I) + myforce

                   ELSE IF ( CG(I) > pwallpQ1(I) ) THEN

                      myeterm = pwallpK(I)*(CG(I)-pwallpQ1(I))* &
                           (CG(I)-pwallpQ1(I))
                      myforce=1.0*2.0*pwallpK(I)*(CG(I)-pwallpQ1(I))

                      ECSUM = ECSUM + myeterm
                      DCH(I) = DCH(I) + myforce

                   ENDIF

                ELSE IF (PTYP == 2) THEN

                   !----------------------------------------------------------------------
                   !  Nth order wall potential

                   IF ((CG(I) >= pwallpa1(I)) .and. &
                        (CG(I) < pwallpb1(I))) THEN

                      term1=(1.0)/((pwallpa1(I)-pwallpb1(I))**3)

                      term2=(CG(I)-pwallpa1(I))**2

                      term2f=CG(I)-pwallpa1(I)
                      term3= 2.0*CG(I) + pwallpa1(I)-3.0*pwallpb1(I)

                      term4=1.0/((pwallpQ1(I)-CG(I)))

                      swchfunc=term1*term2*term3
                      myeterm = pwallpK(I) / &
                           (abs(CG(I)-pwallpQ1(I)))!   **wallpN
                      myeterm = swchfunc*myeterm
                      ECSUM = ECSUM + myeterm

                      myforce = term2f + term3 +  &
                           (0.5*wallpN*term2f*term3*term4)
                      myforce = myforce*2.0*term1*term2f* &
                           pwallpK(I)*(term4**wallpN)

                      DCH(I) = DCH(I) + myforce

                   ELSE IF ((CG(I) >= pwallpb1(I))  .and. &
                        (CG(I) <= pwallpQ1(I)))          THEN
                      myeterm = pwallpK(I)/(abs(CG(I) - &
                           pwallpQ1(I)))**wallpN
                      ECSUM = ECSUM + myeterm

                      myforce = wallpN*pwallpK(I) / &
                           ((pwallpQ1(I)-CG(I))**(wallpN+1))
                      DCH(I) = DCH(I) + myforce


                   ELSE IF (CG(I) > pwallpQ1(I)) THEN

                      myeterm=pwallpK(I) / &
                           (abs(CG(I)-pwallpQ1(I)))**wallpN
                      myforce=-1.0*wallpN*pwallpK(I) / &
                           ((CG(I)-pwallpQ1(I))**(wallpN+1))

                      ECSUM = ECSUm + myeterm
                      DCH(I) = DCH(I) + myforce

                   ELSE IF ((CG(I) <= pwallpa2(I))  .and. &
                        (CG(I) > pwallpb2(I)))        THEN

                      term1=(1.0)/((pwallpa2(I)-pwallpb2(I))**3)
                      term2=(CG(I)-pwallpa2(I))**2
                      term2f=CG(I)-pwallpa2(I)
                      term3= 2.0*CG(I) + pwallpa2(I)- 3.0*pwallpb2(I)
                      term4= 1.0/(CG(I)-pwallpQ2(I))
                      swchfunc=term1*term2*term3
                      myeterm = pwallpK(I) / &
                           (abs(CG(I)-pwallpQ2(I)))**wallpN
                      myeterm = swchfunc*myeterm
                      ECSUM = ECSUM + myeterm

                      myforce = term2f + term3 -  &
                           (0.5*wallpN*term2f*term3*term4)
                      myforce = myforce*2.0*term1*term2f*pwallpK(I) * &
                           (term4**wallpN)
                      DCH(I) = DCH(I) + myforce

                   ELSE IF ((CG(I) <= pwallpb2(I))  .and. &
                        (CG(I) >= pwallpQ2(I)))         THEN

                      myeterm = pwallpK(I) / &
                           (abs(CG(I)-pwallpQ2(I)))**wallpN
                      ECSUM = ECSUM + myeterm

                      myforce=-1.0*wallpN*pwallpK(I) / &
                           ((CG(I)-pwallpQ2(I))**(wallpN+1))
                      DCH(I)=DCH(I) + myforce

                   ELSE IF (CG(I) < pwallpQ2(I)) THEN

                      myeterm =  pwallpK(I) / &
                           (abs(CG(I)-pwallpQ2(I)))**wallpN
                      myforce = wallpN*pwallpK(I) / &
                           ((pwallpQ2(I)-CG(I))**(wallpN+1))
                      ECSUM = ECSUM + myeterm
                      DCH(I) = DCH(I) + myforce

                   ENDIF
                ENDIF

                !---------------------------------------------------------------------
                !_____________________________________________________________________
             ENDIF
          ENDDO

          ! off-diagonal terms for hardness term
          DO I=1,NATOM-1
             IF (QPARTBIN(I) /= 0) THEN
                !     General case
                IF (MOLTYPE(I) == 0) THEN
                   IF (I == 1) THEN
                      NBINDX=1
                   ELSE
                      NBINDX=IBLO14(I-1)+1
                   ENDIF
                   JCOUNT=0
                   JCTR=IBLO14(I)
                   CGT=CCELEC*CG(I)
                   IF (CGT /= ZERO) THEN
                      CRXI=X(I)
                      CRYI=Y(I)
                      CRZI=Z(I)
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN
                            JCOUNT=JCOUNT+1
                            CGTJ=CCELEC*CG(J)
                            DXIJ=CRXI-X(J)
                            DYIJ=CRYI-Y(J)
                            DZIJ=CRZI-Z(J)
                            S=DXIJ*DXIJ+DYIJ*DYIJ+DZIJ*DZIJ
                            HIJ1=EHA(I)+EHA(J)
                            HIJ=1.00*HIJ1/SQRT((1.0+HIJ1*HIJ1*S)) ! SCALED OFF DIAG. ETA
                            ECSUM1=ECSUM1+2*CG(I)*HIJ*CG(J)*CCELEC
                            DCH(I)=DCH(I)+HIJ*CG(J)*CCELEC
                            DCH(J)=DCH(J)+HIJ*CG(I)*CCELEC
                            IF(QREF) THEN
                               DCH(I)=DCH(I)+HIJ*CGREF(J)
                               DCH(J)=DCH(J)+HIJ*CGREF(I)
                            ENDIF
                            !     FORCE FROM CHARGE FLUCTUATING ON NUCLEI
                            IF (QCGMINE.OR.(.NOT.QCGWATER)) THEN
                               FI=CG(J)*HIJ*HIJ*HIJ
                               DX(I)=DX(I)-FI*DXIJ*CGT
                               DY(I)=DY(I)-FI*DYIJ*CGT
                               DZ(I)=DZ(I)-FI*DZIJ*CGT
                               FJ=CG(I)*HIJ*HIJ*HIJ
                               DX(J)=DX(J)+FJ*DXIJ*CGTJ
                               DY(J)=DY(J)+FJ*DYIJ*CGTJ
                               DZ(J)=DZ(J)+FJ*DZIJ*CGTJ
                            ENDIF
                         ELSE
                         ENDIF
                      ENDDO   !JJ
                   ELSE
                      CRXI=X(I)
                      CRYI=Y(I)
                      CRZI=Z(I)
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN
                            CGTJ=CCELEC*CG(J)
                            DXIJ=CRXI-X(J)
                            DYIJ=CRYI-Y(J)
                            DZIJ=CRZI-Z(J)
                            S=DXIJ*DXIJ+DYIJ*DYIJ+DZIJ*DZIJ
                            HIJ=EHA(I)+EHA(J)
                            HIJ=HIJ/SQRT((1.0+HIJ*HIJ*S))
                            DCH(I)=DCH(I)+HIJ*CG(J)*CCELEC
                            IF(QREF) THEN
                               DCH(I)=DCH(I)+HIJ*CGREF(J)
                               DCH(J)=DCH(J)+HIJ*CGREF(I)
                            ENDIF
                            IF (QCGMINE.OR.(.NOT.QCGWATER) .AND. &
                                 (CGTJ /= ZERO))                THEN
                               FJ=CG(I)*HIJ*HIJ*HIJ
                               DX(J)=DX(J)+FJ*DXIJ*CGTJ
                               DY(J)=DY(J)+FJ*DYIJ*CGTJ
                               DZ(J)=DZ(J)+FJ*DZIJ*CGTJ
                            ENDIF
                         ENDIF
                      ENDDO   !JJ
                   ENDIF      !(CGT /= ZERO)
                ELSE IF ( (MOLTYPE(I) > 0)  .AND. &
                     (MOLTYPE(I) <= 2))     THEN ! for SPC water molecules
                   CGT=CCELEC*CG(I)
                   IF (I == 1) THEN
                      NBINDX=1
                   ELSE
                      NBINDX=IBLO14(I-1)+1
                   ENDIF
                   JCTR=IBLO14(I)
                   IF (CGT /= ZERO) THEN
                      !     CRXI=X(I)
                      !     CRYI=Y(I)
                      !     CRZI=Z(I)
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN
                            IF (MOLTYPE(I) == 1 .OR. &
                                 MOLTYPE(J) == 1)   THEN
                               HIJ = 276.0
                            ELSE
                               HIJ = 196.0
                            ENDIF
                            ECSUM1=ECSUM1+2*CG(I)*HIJ*CG(J)
                            DCH(I)=DCH(I)+HIJ*CG(J)
                            DCH(J)=DCH(J)+HIJ*CG(I)
                            IF(QREF) THEN
                               DCH(I)=DCH(I)+HIJ*CGREF(J)
                               DCH(J)=DCH(J)+HIJ*CGREF(I)
                            ENDIF
                         ENDIF
                      ENDDO   !JJ
                      NBINDX=NBINDX+IBLO14(I)
                   ELSE
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN
                            S=(X(I)-X(J))*(X(I)-X(J)) &
                                 + (Y(I)-Y(J))*(Y(I)-Y(J)) &
                                 +(Z(I)-Z(J))*(Z(I)-Z(J))
                            !     The type stuff is for SPC test only
                            IF (MOLTYPE(I) == 1) THEN
                               HIJ=276.0
                            ELSE IF (ITC(IAC(J)) == ITC(IAC(1))) THEN
                               HIJ=276.0
                            ELSE
                               HIJ=196.0
                            ENDIF
                            ECSUM1=ECSUM1+2*CG(I)*HIJ*CG(J)
                            DCH(I)=DCH(I)+HIJ*CG(J)
                            IF(QREF) THEN
                               DCH(I)=DCH(I)+HIJ*CGREF(J)
                               DCH(J)=DCH(J)+HIJ*CGREF(I)
                            ENDIF
                         ENDIF
                      ENDDO   !JJ
                   ENDIF
                   !     
                   !     TIP4P STUFF
                ELSE IF ( (MOLTYPE(I) > 2) .AND. &
                     (MOLTYPE(I) <= 9))   THEN ! for TIP4P water molecules
                   CGT=CCELEC*CG(I)
                   IF (I == 1) THEN
                      NBINDX=1
                   ELSE
                      NBINDX=IBLO14(I-1)+1
                   ENDIF
                   JCTR=IBLO14(I)
                   IF (CGT /= ZERO) THEN
                      !     CRXI=X(I)
                      !     CRYI=Y(I)
                      !     CRZI=Z(I)
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN

                            IF (MOLTYPE(J) /= 9) THEN
                               IF (MOLTYPE(I) == 3 .OR. &
                                    MOLTYPE(J) == 3) THEN
                                  HIJ = 286.4
                               ELSE
                                  HIJ = 203.6
                               ENDIF
                               ECSUM1=ECSUM1+2*CG(I)*HIJ*CG(J)
                               DCH(I)=DCH(I)+HIJ*CG(J)
                               DCH(J)=DCH(J)+HIJ*CG(I)
                               IF(QREF) THEN
                                  DCH(I)=DCH(I)+HIJ*CGREF(J)
                                  DCH(J)=DCH(J)+HIJ*CGREF(I)
                               ENDIF
                            ENDIF
                         ENDIF
                      ENDDO   !JJ
                   ENDIF

                ELSE          ! Other Water
                   IF (I == 1) THEN
                      NBINDX=1
                   ELSE
                      NBINDX=IBLO14(I-1)+1
                   ENDIF
                   JCTR=IBLO14(I)
                   CGT=CCELEC*CG(I)
                   IF (CGT /= ZERO) THEN
                      CRXI=X(I)
                      CRYI=Y(I)
                      CRZI=Z(I)
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN
                            DXIJ=CRXI-X(J)
                            DYIJ=CRYI-Y(J)
                            DZIJ=CRZI-Z(J)
                            S=DXIJ*DXIJ+DYIJ*DYIJ+DZIJ*DZIJ
                            HIJ1=EHA(I)+EHA(J)
                            HIJ=HIJ1/SQRT((1.0+HIJ1*HIJ1*S))
                            ECSUM1=ECSUM1+2*CG(I)*HIJ*CG(J)*CCELEC
                            DCH(I)=DCH(I)+HIJ*CG(J)*CCELEC
                            DCH(J)=DCH(J)+HIJ*CG(I)*CCELEC
                            IF(QREF) THEN
                               DCH(I)=DCH(I)+HIJ*CGREF(J)*CCELEC
                               DCH(J)=DCH(J)+HIJ*CGREF(I)*CCELEC
                            ENDIF
                         ENDIF
                      ENDDO   !JJ
                   ELSE
                      DO JJ=NBINDX,JCTR
                         J=IABS(INB14(JJ))
                         IF (J > 0) THEN
                            S=(X(I)-X(J))*(X(I)-X(J)) &
                                 +(Y(I)-Y(J))*(Y(I)-Y(J)) &
                                 +(Z(I)-Z(J))*(Z(I)-Z(J))
                            HIJ=EHA(I)+EHA(J)
                            HIJ=HIJ/SQRT((1.0+HIJ*HIJ*S))
                            ECSUM1=ECSUM1+2*CG(I)*HIJ*CG(J)*CCELEC
                            DCH(I)=DCH(I)+HIJ*CG(J)*CCELEC
                            IF(QREF) THEN
                               DCH(I)=DCH(I)+HIJ*CGREF(J)*CCELEC
                            ENDIF
                         ENDIF
                      ENDDO   !JJ
                   ENDIF      !(CGT /= ZERO)
                ENDIF         ! MOLTYPE
             ENDIF            ! QPARTBIN
          ENDDO
       ELSE                   !(CGMODEL ==  (anything but 0 or 1) )
          WRITE(OUTU,'(A,I5)')"CGMODEL out of range: ",CGMODEL
          CALL DIE
       ENDIF                  !(CGMODEL == 
       ECSUM=(ECSUM+0.5*ECSUM1)
       IF (CGMODEL >= 1) THEN
          !     don't double count the simple coulombic interactions
          ETERM(ELEC)=ETERM(ELEC)+ECSUM
          EPROP(CGPOT)=ETERM(ELEC)
          IF(LEWALD.AND.LELEC.AND.QETERM(ELEC)) &
               EPROP(CGPOT)=EPROP(CGPOT)+ETERM(IMELEC)
          call set_param('ESEL',ECSUM)
       ENDIF


#if KEY_PARALLEL==1
    ELSE                      ! (IF MYNOD == 0)
       EPROP(CGPOT)=ETERM(ELEC)
       IF(LEWALD.AND.LELEC.AND.QETERM(ELEC)) &
            EPROP(CGPOT)=EPROP(CGPOT)+ETERM(IMELEC)

    ENDIF                     ! (IF MYNOD == 0)

#endif /*                          (IF PARALLEL)*/

69  format(22(f10.5,3x))
    IFLAG = 0


    RETURN
  END  SUBROUTINE QBENER

  SUBROUTINE QNOCOD(DX,DY,DZ,NATOM)
    !-----------------------------------------------------------------------
    !  This subroutine will set all atom first derivatives (or any three
    !  arrays of length NATOM) to zero.

    use number
    INTEGER I,NATOM
    real(chm_real) DX(*),DY(*),DZ(*)
    !
    DO I=1,NATOM
       DX(I)=ZERO
       DY(I)=ZERO
       DZ(I)=ZERO
    ENDDO
    RETURN
  END   SUBROUTINE QNOCOD

  SUBROUTINE DDCG(BNBND,NATOM,CG,X,Y,Z)
    !-----------------------------------------------------------------------
    !
    use inbnd
    use stream
    !---   use nbutil_module,only:getbnd
    !!    integer bnbnd(*)
    type(nonbondDataStructure) bnbnd
    integer natom
    real(chm_real)  cg(*)
    real(chm_real)  x(*),y(*),z(*)
    real(chm_real) hij
    !
    if(natom <= 0) return
    !     make sure we have the current non-bond flags and cut-offs. this
    !     will handle any changes made due to pert or tsm.
    call getbnd(bnbnd,.true.)
    !
    call nbddcg(natom,bnbnd%jnb,bnbnd%inblo, &
         bnbnd%iblo14,bnbnd%inb14,cg,x,y,z,ctofnb)
    return
  end subroutine ddcg

  !-----------------------------------------------------------------------
  !                NBDDCG
  !-----------------------------------------------------------------------
  subroutine nbddcg(natom,jnb,inblo,iblo14,inb14,cg,x,y,z,ctofnb)
    !-----------------------------------------------------------------------
    !  need to check these derivatives.  I'm not sure they are correct.
    !
    !     NATOM  - number of atoms
    !     JNB    - nonbond pair list  (INBLO(NATOM))
    !     INBLO  - pointers into JNB  (NATOM)
    !     CG     - charges  (NATOM)
    !

    real(chm_real) HIJ
    !
    INTEGER NATOM
    INTEGER JNB(*)
    INTEGER INBLO(*)
    INTEGER IBLO14(*)
    INTEGER INB14(*)
    real(chm_real) CG(*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) CTOFNB
    !
    real(chm_real) C2OFNB
    real(chm_real) CRXI,CRYI,CRZI,DXI,DYI,DZI
    real(chm_real) S
    INTEGER I,J,I1,J1,J2,NB,NPR,JPR,ITEMP
    INTEGER NBINDX,JJ,JCTR
    !
    DO I=1,NATOM
       DDCH(I)=0.0
    ENDDO
    NB=0
    C2OFNB=CTOFNB*CTOFNB
    !
    ITEMP=0
    loop40: DO I=1,NATOM
       NPR=INBLO(I)-ITEMP
       ITEMP=INBLO(I)
       IF (NPR == 0) cycle loop40
       !
       CRXI=X(I)
       CRYI=Y(I)
       CRZI=Z(I)
       !
       DO JPR=1,NPR
          NB=NB+1
          IF (JNB(NB) < 0) THEN
             J=-JNB(NB)
          ELSE
             J=JNB(NB)
          ENDIF
          !
          DXI=CRXI-X(J)
          DYI=CRYI-Y(J)
          DZI=CRZI-Z(J)
          S=DXI*DXI+DYI*DYI+DZI*DZI
          !
          IF (S < C2OFNB) THEN
             HIJ=1.0/SQRT(S)
             DDCH(I)=DDCH(I)+HIJ*DCH(J)
             DDCH(J)=DDCH(J)+HIJ*DCH(I)
          ENDIF
          !
       enddo
    enddo loop40
    !
    ! Calculate the contributions to DDCH from nonbond exclusion list
    NBINDX=1
    DO I=1,NATOM-1
       IF (I == 1) THEN
          NBINDX=1
       ELSE
          NBINDX=IBLO14(I-1)+1
       ENDIF
       JCTR=IBLO14(I)
       DO JJ=NBINDX,JCTR
          J=IABS(INB14(JJ))  !!! SAP
          !          J=INB14(JJ)
          S=(X(I)-X(J))*(X(I)-X(J))+(Y(I)-Y(J))*(Y(I)-Y(J)) &
               +(Z(I)-Z(J))*(Z(I)-Z(J))
          HIJ=EHA(I)+EHA(J)
          HIJ=HIJ/SQRT((1.0+HIJ*HIJ*S))
          DDCH(I)=DDCH(I)+HIJ*DCH(J)
          !          DDCH(J)=DDCH(J)+HIJ*DCH(I)
       ENDDO
    ENDDO
    DO I=1,NATOM
       !        DDCH(I)=-2.0*DDCH(I)
       DDCH(I)=2.0*DDCH(I)
    ENDDO

    !      WRITE(OUTU,*)(DCH(I),DDCH(I),I=1,NATOM)

    RETURN
  END SUBROUTINE NBDDCG

  SUBROUTINE CGINV(NATOM,X,Y,Z,DX,DY,DZ,CG,ECH,EHA,QMOL, &
       MOLT,DA,B,QCGINVF,JNB,INBLO,INB14,IBLO14,IAC,QPARTBIN, &
       MOLTYPE,QPRNETA,CCNBA,CCNBB,CCNBC,CCNBD,IACNB,NITCC2,LOWTP, &
#if KEY_WCA==1
       LLSOFT,RCUT,WCA, &   
#endif
       OUTU)
    !----------------------------------------------------------------------
#if KEY_FLUCQ==1
    use flucqm,only:fqcfor             
#endif

    use consta
    use number
#if KEY_FLUCQ==1
    use flucq 
#endif

    INTEGER NB,ITEMP,NATOM,JNB(*),INB14(*),INBLO(*),IBLO14(*)
    real(chm_real) X(*),Y(*),Z(*),CG(*),ECH(*),EHA(*),B(*)
    real(chm_real) U(6,6),V(6)
    real(chm_real) DX(*),DY(*),DZ(*),DA(NATOM,*)
    INTEGER N,MOLT(*),MOLTYPE(*)
    INTEGER OUTU
#if KEY_BLOCK==1
    ! will need to change these declarations to get block working with
    ! charge inversion
    INTEGER IBLOCK(1)
    real(chm_real)  BLCOE(1),BLCOV(1),BLCOVR(1),BLCOVA(1)
#endif /*  BLOCK*/
    real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
    INTEGER IACNB(*),NITCC2,LOWTP(*)
    LOGICAL QMOL,QCGINVF,QPRNETA
    real(chm_real) RIJ2,RIJ,HIJ,RIJ1,HIJ1
    INTEGER QPARTBIN(*),IAC(*),NPR,JPR,JCTR,JJ,NBINDX
    real(chm_real) EVDW,ECG
    LOGICAL LUSED

#if KEY_WCA==1
    LOGICAL LLSOFT
    real(chm_real) RCUT
    real(chm_real) WCA(*)
#endif 
    !
    !       DIMENSION INDX(6)
    INTEGER I,J,K
    INTEGER NMOL
    integer :: alloc_err
    real(chm_real),allocatable,dimension(:,:) :: a

    allocate(a(natom,natom),stat=alloc_err)
    if(alloc_err /= 0) &
         CALL WrnDie(-1,'<cheqmodule.src>FQPOLTEN', &
         'Failed to deallocate memory for A array')

    DO I=1,NATOM
       DO J=1,NATOM
          A(I,J)=0.0
       ENDDO
       IF (MOLTYPE(I) == 0) THEN
          A(I,I)=2.0*EHA(I)
       ELSE IF (MOLTYPE(I) > 0) THEN  ! for SPC water molecules
          IF (MOLTYPE(I) == 1) THEN
             A(I,I)=367.0/CCELEC
          ELSE
             A(I,I)=392.2/CCELEC
          ENDIF
       ENDIF
    ENDDO
    !

    CALL ENBFS8(EVDW,ECG,.TRUE.,.FALSE.,1,NATOM,CG,JNB,INBLO, &
         IACNB, NITCC2,LOWTP, &
#if KEY_BLOCK==1
         IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA,  & 
#endif
         A,.TRUE., &
#if KEY_FLUCQ==1
         QFLUC,FQCFOR,         & 
#endif
#if KEY_WCA==1
         LLSOFT,RCUT,WCA,      & 
#endif
         LUSED, .false.)

    NBINDX=1
    DO I=1,NATOM-1
       IF (MOLTYPE(I) == 0) THEN
          IF (I == 1) THEN
             NBINDX=1
          ELSE
             NBINDX=IBLO14(I-1)+1
          ENDIF
          JCTR=IBLO14(I)
          DO JJ=NBINDX,JCTR
             J=IABS(INB14(JJ))
             IF (J > 0) THEN
                RIJ2=(X(I)-X(J))*(X(I)-X(J)) &
                     +(Y(I)-Y(J))*(Y(I)-Y(J)) &
                     +(Z(I)-Z(J))*(Z(I)-Z(J))
                HIJ=EHA(I)+EHA(J)
                HIJ=HIJ/SQRT(1.0+HIJ*HIJ*RIJ2)
                A(I,J)=HIJ
                A(J,I)=HIJ

             ELSE
                HIJ = 1.0/SQRT(RIJ2)
                A(I,J) = HIJ
                A(J,I) = HIJ
             ENDIF

          ENDDO
       ELSE IF (MOLTYPE(I) > 0) THEN  ! for SPC water molecules
          IF (I == 1) THEN
             NBINDX=1
          ELSE
             NBINDX=IBLO14(I-1)+1
          ENDIF
          JCTR=IBLO14(I)
          DO JJ=NBINDX,JCTR
             J=IABS(INB14(JJ))
             IF (J > 0) THEN
                IF (MOLTYPE(I) == 1.OR.MOLTYPE(J).EQ.1) THEN
                   A(I,J)=276.0/CCELEC
                   A(J,I)=276.0/CCELEC
                ELSE
                   A(I,J)=196.0/CCELEC
                   A(J,I)=196.0/CCELEC
                ENDIF
             ENDIF
          ENDDO !JJ
          NBINDX=NBINDX+IBLO14(I)
       ELSE
          write(6,*) 'CGINV: other waters not supported yet'
       ENDIF
    ENDDO

    IF(QPRNETA) THEN
       DO I=1,NATOM
          write(6,*) 'Row ',I
          WRITE(6,60) (A(I,J), J=1,NATOM)
       ENDDO
    ENDIF
60  FORMAT(12(F10.5,1x))

    IF (QMOL) THEN
       ! assumes partitions are consecutive!!!
       NMOL=0
       DO I=2,NATOM
          IF (QPARTBIN(I) /= QPARTBIN(I-1)) NMOL=NMOL+1
       ENDDO
       K=1
       B(1)=CG(1)
       DO I=2,NATOM
          IF (QPARTBIN(I) == QPARTBIN(I-1)) THEN
             DO J=1,NATOM
                A(I,J)=A(K,J)-A(I,J)
             ENDDO
             B(I)=(ECH(I)-ECH(K))/CCELEC
             B(K)=B(K)+CG(I)
          ELSE
             K=I
             B(I)=CG(I)
          ENDIF
       ENDDO
       DO J=1,NATOM
          IF(QPARTBIN(J) == QPARTBIN(1)) THEN
             A(1,J)=1.0
          ELSE
             A(1,J)=0.0
          ENDIF
       ENDDO
       DO I=2,NATOM
          IF (QPARTBIN(I) /= QPARTBIN(I-1)) THEN
             DO J=1,NATOM
                IF(QPARTBIN(J) == QPARTBIN(I)) THEN
                   A(I,J)=1.0
                ELSE
                   A(I,J)=0.0
                ENDIF
             ENDDO
          ENDIF
       ENDDO
    ELSE
       DO I=2,NATOM
          DO J=1,NATOM
             A(I,J)=A(1,J)-A(I,J)
          ENDDO
       ENDDO
       DO I=1,NATOM
          B(I)=(ECH(I)-ECH(1))/CCELEC
          A(1,I)=1.0
       ENDDO
    ENDIF
    IF(QPRNETA) THEN
       write(6,*) 'A matrix going into GAUSSJ'
       DO I=1,NATOM
          write(6,*) 'Row ',I
          WRITE(6,60) (A(I,J), J=1,NATOM)
       ENDDO
    ENDIF
    QPRNETA=.FALSE.
    CALL GAUSSJ(A,NATOM,NATOM,B,1,1)
    ! So A is the inverse matrix now

    RETURN
  END SUBROUTINE CGINV

  SUBROUTINE GAUSSJ(A,N,NP,B,M,MP)
    !----------------------------------------------------------------------
    use number
    !
    INTEGER N,NP,M,MP
    real(chm_real) A(NP,NP),B(NP,MP)
    !
    INTEGER, PARAMETER :: NMAX=1000
    INTEGER IPIV(NMAX),INDXR(NMAX),INDXC(NMAX)
    !
    INTEGER I,J,K,L,LL,IROW,ICOL
    real(chm_real)  BIG,DUM,PIVINV
    !
    IF(N > NMAX) CALL WRNDIE(-4,'<GAUSSJ>','N is too large.')
    !
    DO J=1,N
       IPIV(J)=0
    ENDDO
    DO I=1,N
       BIG=ZERO
       DO J=1,N
          IF(IPIV(J) /= 1)THEN
             DO K=1,N
                IF (IPIV(K) == 0) THEN
                   IF (ABS(A(J,K)) >= BIG)THEN
                      BIG=ABS(A(J,K))
                      IROW=J
                      ICOL=K
                   ENDIF
                ELSE IF (IPIV(K) > 1) THEN
                   CALL WRNDIE(-4,'<GAUSSJ>','Singular matrix.1')
                ENDIF
             ENDDO
          ENDIF
       ENDDO
       IPIV(ICOL)=IPIV(ICOL)+1
       IF (IROW /= ICOL) THEN
          DO L=1,N
             DUM=A(IROW,L)
             A(IROW,L)=A(ICOL,L)
             A(ICOL,L)=DUM
          ENDDO
          DO L=1,M
             DUM=B(IROW,L)
             B(IROW,L)=B(ICOL,L)
             B(ICOL,L)=DUM
          ENDDO
       ENDIF
       INDXR(I)=IROW
       INDXC(I)=ICOL
       IF (A(ICOL,ICOL) == 0.)  &
            CALL WRNDIE(-4,'<GAUSSJ>','Singular matrix.2')
       PIVINV=1./A(ICOL,ICOL)
       A(ICOL,ICOL)=1.
       DO L=1,N
          A(ICOL,L)=A(ICOL,L)*PIVINV
       ENDDO
       DO L=1,M
          B(ICOL,L)=B(ICOL,L)*PIVINV
       ENDDO
       DO LL=1,N
          IF(LL /= ICOL)THEN
             DUM=A(LL,ICOL)
             A(LL,ICOL)=0.
             DO L=1,N
                A(LL,L)=A(LL,L)-A(ICOL,L)*DUM
             ENDDO
             DO L=1,M
                B(LL,L)=B(LL,L)-B(ICOL,L)*DUM
             ENDDO
          ENDIF
       ENDDO
    ENDDO
    DO L=N,1,-1
       IF(INDXR(L) /= INDXC(L))THEN
          DO K=1,N
             DUM=A(K,INDXR(L))
             A(K,INDXR(L))=A(K,INDXC(L))
             A(K,INDXC(L))=DUM
          ENDDO
       ENDIF
    ENDDO
    RETURN
  END  SUBROUTINE GAUSSJ

  SUBROUTINE CGCOPY(CG,CGX,NATOM,NATIM)
    !----------------------------------------------------------------------
    real(chm_real) CG(*),CGX(*)
    INTEGER NATOM,NATIM,I,NATX
    IF (NATIM > NATOM) THEN
       NATX=NATIM
    ELSE
       NATX=NATOM
    ENDIF
    DO I=1,NATX
       CGX(I)=CG(I)
    ENDDO
    RETURN
  END SUBROUTINE CGCOPY

  SUBROUTINE CGCPY(CGX)
    !----------------------------------------------------------------------
    use psf
    real(chm_real) CGX(*)
    INTEGER I
    DO I=1,NATOM
       CG(I)=CGX(I)
    ENDDO
    RETURN
  END  SUBROUTINE CGCPY

  SUBROUTINE QCGTRANSI(DCH,NATOM,NATIM,IMATTR)
    !----------------------------------------------------------------------
    use number
    real(chm_real) DCH(*)
    INTEGER NATOM,NATIM,IMATTR(*)
    INTEGER I

    DO I=NATOM+1,NATIM
       DCH(IMATTR(I)) = DCH(IMATTR(I)) + DCH(I)
       DCH(I)=ZERO
    ENDDO

    RETURN
  END SUBROUTINE QCGTRANSI

  SUBROUTINE PRCGIMG(IMATTRX)
    !----------------------------------------------------------------------
    ! THIS ROUTINE COPIES CG FROM MAIN SET TO IMAGE ATOMS
    ! FOR USE IS CHARGE DYNAMICS
    !
    use image
    use psf
    !
    INTEGER IMATTRX(*)
    !
    INTEGER I,J
    IF (NATIM > NATOM) THEN
       WRITE(6,*)"IMAGE CHARGES>>>> comparison"
       DO I=NATOM+1,NATIM
          J=IMATTRX(I)
          write(6,'(2I10,2F20.6)')I,J,CG(I),CG(J)
       ENDDO
    ENDIF
    RETURN
  END  SUBROUTINE PRCGIMG

  SUBROUTINE FQPOLTEN(NATOM,X,Y,Z,DX,DY,DZ,CG,ECH,EHA,QMOL, &
       MOLT,B,QCGINVF,JNB,INBLO,INB14,IBLO14,IAC,QPARTBIN, &
       MOLTYPE,QPRNETA,CCNBA,CCNBB,CCNBC,CCNBD,IACNB,NITCC2,LOWTP, &
       OUTU)
    !------------------------------------------------------------------------
    !    subroutine to calcuate molecualr polarizability tensor
    !    from atomic hardness parameters
    !   USE ONLY FOR SINGLE MOLECULES WITH NET ZERO CHARGE!!!!!
    !    (modify normalization scheme for use with charged species
    !    or more exotic normalization schemes
    use consta
    INTEGER NB,ITEMP,NATOM,JNB(*),INB14(*),INBLO(*),IBLO14(*)
    real(chm_real) X(*),Y(*),Z(*),CG(*),ECH(*),EHA(*),B(*)
    real(chm_real) ISICK(3,3),BSICK(3,1)
    real(chm_real) U(6,6),V(6)
    real(chm_real) DX(*),DY(*),DZ(*)
    INTEGER N,MOLT(*),MOLTYPE(*)
    INTEGER OUTU,KK,LL,JJ,NATMM1,L,M,TIPFLAG
    real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
    INTEGER IACNB(*),NITCC2,LOWTP(*)
    LOGICAL QMOL,QCGINVF,QPRNETA
    real(chm_real) RIJ2,RIJ,HIJ,RIJ1,HIJ1
    INTEGER QPARTBIN(*),IAC(*),NPR,JPR,JCTR,NBINDX
    real(chm_real) EVDW,ECG, alpha(3,3)    ! ,inter(3,NATOM),R(3,NATOM)
    !     real(chm_real) delR(NATOM,3), temp(3,NATOM)
    LOGICAL LUSED
    Character(len=2) comp
    !
    INTEGER I,J,K
    INTEGER NMOL
    integer :: alloc_err
    real(chm_real),allocatable,dimension(:,:) :: a,inter,r,delr,temp

    allocate(A(NATOM,natom),inter(3,natom), &
         r(3,natom),delr(natom,3), &
         stat=alloc_err)
    if(alloc_err /= 0) &
         CALL WrnDie(-1,'<cheqmodule.src>FQPOLTEN', &
         'Failed to deallocate memory for A array')


    TIPFLAG=0

    !       set up the eta matrix

    DO I=1,NATOM
       DO J=1,NATOM
          A(I,J)=0.0
       ENDDO
    ENDDO

    DO I=1,3
       DO J=1,3
          ISICK(I,J) = 0.0
          alpha(I,J) = 0.0
       ENDDO
    ENDDO

    DO I = 1,3
       DO J = 1,NATOM
          inter(I,J) = 0.0
       ENDDO
    ENDDO

    DO I=1,NATOM
       DO J=1,NATOM
          A(I,J)=0.0
       ENDDO
       IF (MOLTYPE(I) == 0) THEN
          A(I,I)=2.0*EHA(I)
       ELSE IF (MOLTYPE(I) > 0) THEN ! for SPC / TIP4P water molecules
          IF (MOLTYPE(I) == 1) THEN
             A(I,I)=367.0/CCELEC
          ELSEIF (MOLTYPE(I) == 2) THEN
             A(I,I)=392.2/CCELEC
          ELSEIF (MOLTYPE(I) == 3) THEN
             A(I,I) = 371.6/CCELEC !  TIP4P OM SITE
          ELSEIF (MOLTYPE(I) == 4) THEN
             A(I,I) = 353.0/CCELEC ! TIP4P HYDROGEN SITE
          ELSEIF (MOLTYPE(I) == 9) THEN
             A(I,I) = 0.0    !   TIP4P oxygen site (no charge)
          ELSE
          ENDIF
       ENDIF
    ENDDO

    NBINDX=1
    DO I=1,NATOM-1
       IF (MOLTYPE(I) == 0) THEN
          IF (I == 1) THEN
             NBINDX=1
          ELSE
             NBINDX=IBLO14(I-1)+1
          ENDIF
          JCTR=IBLO14(I)
          DO JJ=NBINDX,JCTR

             J=IABS(INB14(JJ)) !!! SAP
             IF (J > 0) THEN
                RIJ2=(X(I)-X(J))*(X(I)-X(J)) &
                     +(Y(I)-Y(J))*(Y(I)-Y(J)) &
                     +(Z(I)-Z(J))*(Z(I)-Z(J))
                HIJ=EHA(I)+EHA(J)
                HIJ=HIJ/SQRT(1.0+HIJ*HIJ*RIJ2)
                A(I,J)=HIJ
                A(J,I)=HIJ
             ELSE
                HIJ = 1.0/SQRT(RIJ2)
                A(I,J) = HIJ
                A(J,I) = HIJ
             ENDIF

          ENDDO

       ELSE IF ( (MOLTYPE(I) > 0).AND.(MOLTYPE(I) <= 2) ) THEN
          !     for SPC/TIP4P water molecules
          IF (I == 1) THEN
             NBINDX=1
          ELSE
             NBINDX=IBLO14(I-1)+1
          ENDIF
          JCTR=IBLO14(I)
          DO JJ=NBINDX,JCTR
             J=IABS(INB14(JJ))
             IF (J > 0) THEN
                IF (MOLTYPE(I) == 1.OR.MOLTYPE(J).EQ.1) THEN
                   A(I,J)=276.0/CCELEC
                   A(J,I)=276.0/CCELEC
                ELSE
                   A(I,J)=196.0/CCELEC
                   A(J,I)=196.0/CCELEC
                ENDIF
             ENDIF
          ENDDO              !JJ
          NBINDX=NBINDX+IBLO14(I)

       ELSE IF ( (MOLTYPE(I) > 2).AND.(MOLTYPE(I) < 9)) THEN
          !     for TIP4P water molecu
          TIPFLAG=1
          IF (I == 1) THEN
             NBINDX=1
          ELSE
             NBINDX=IBLO14(I-1)+1
          ENDIF
          JCTR=IBLO14(I)
          !     CRXI=X(I)
          !     CRYI=Y(I)
          !     CRZI=Z(I)
          DO JJ=NBINDX,JCTR
             J=IABS(INB14(JJ)) !!! SAP
             !     J=INB14(JJ)
             IF (J > 0) THEN

                IF (MOLTYPE(J) /= 9)  THEN
                   IF (MOLTYPE(I) == 3.OR.MOLTYPE(J).EQ.3) THEN
                      A(I,J) = 286.4/CCELEC
                      A(J,I) = 286.4/CCELEC
                   ELSE
                      A(I,J) = 203.6/CCELEC
                      A(J,I) = 203.6/CCELEC
                   ENDIF
                ENDIF
             ENDIF
          ENDDO              !JJ
          !     ENDIF

       ELSE
          write(6,*) 'CGINV: other waters not supported yet'
       ENDIF
    ENDDO

    IF (TIPFLAG == 1) THEN
       DO L=1,3
          DO M=1,3
             ISICK(L,M) = A(L+1,M+1)
          ENDDO
       ENDDO
    ENDIF

    !     compute inverse of eta matrix

    IF (TIPFLAG == 0) THEN
       CALL GAUSSJ(A,NATOM,NATOM,B,1,1)
    ELSEIF (TIPFLAG == 1) THEN
       CALL GAUSSJ(ISICK,3,3,BSICK,1,1)
    ENDIF

    !     A is the inverse matrix now

    !     now set up the coordinates matrix   `R`

    IF (TIPFLAG == 0) THEN
       DO JJ=1,NATOM
          R(1,JJ)= X(JJ)
          R(2,JJ)= Y(JJ)
          R(3,JJ)= Z(JJ)
       ENDDO

    ELSE

       DO JJ=1,3
          R(1,JJ) = X(JJ+1)
          R(2,JJ) = Y(JJ+1)
          R(3,JJ) = Z(JJ+1)
       ENDDO

    ENDIF

    !     now set up the delR matrix

    DO JJ=1,3
       DO KK=1,NATOM
          temp(JJ,KK)=R(JJ,KK)-R(JJ,1)
          delR(KK,JJ)=temp(JJ,KK)
       ENDDO
    ENDDO

    IF (TIPFLAG == 0) THEN

       !     multiply  R*inv(eta)

       DO JJ=1,3
          DO KK=1,NATOM
             DO LL=1,NATOM
                inter(JJ,KK)  = inter(JJ,KK)+R(JJ,LL)*A(LL,KK)
             ENDDO
          ENDDO
       ENDDO

       DO JJ=1,3
          DO KK=1,3
             DO LL=1,NATOM
                alpha(JJ,KK) = alpha(JJ,KK)+inter(JJ,LL)*R(KK,LL)
             ENDDO
          ENDDO
       ENDDO

    ELSEIF (TIPFLAG == 1) THEN

       !     multiply  R*inv(eta)

       DO JJ=1,3
          DO KK=1,3          ! NATOM
             DO LL=1,3       ! NATOM
                inter(JJ,KK) = inter(JJ,KK) + R(JJ,LL)*ISICK(LL,KK)
             ENDDO
          ENDDO
       ENDDO

       DO JJ=1,3
          DO KK=1,3
             DO LL=1,3       ! NATOM
                alpha(JJ,KK)  = alpha(JJ,KK)+inter(JJ,LL)*R(KK,LL)
             ENDDO
          ENDDO
       ENDDO
    ENDIF

    write(OUTU,*) " "
    write(OUTU,*)" MOLECULAR POLARIZABILITY TENSOR ", TIPFLAG
    write(OUTU,*) "          X            Y            Z "
    write(OUTU,*) "  _____________________________________"
    DO I = 1,3
       if (I == 1) comp='X|'
       if (I == 2) comp='Y|'
       if (I == 3) comp='Z|'
       write(6,22) comp,(alpha(I,J),J=1,3)
    ENDDO
    write(OUTU,*) " "
    deallocate(a)

22  format    (1x,a2,2x,3(f10.5,2x))
23  format    (4(f10.5,2x))


    return
  end  SUBROUTINE FQPOLTEN

  !  routine to get forces on charges arising from the GB model for solvation

  SUBROUTINE  GBFONCHEQ(X,Y,Z)
    !-----------------------------------------------------------------------
    !     This subroutine calculates the forces on charges arising from the
    !     generalized born expression for the solvation energy.  Coming into
    !     the routine, the Born radii, alpha(i), should be calculated, of if
    !     needed, updated in a multiple time step approach, and available.

    use gb_common,only:alph_gb,eps_gb

    use consta
    use number
    use psf
    use inbnd
    use parallel

    Integer I, J
    real(chm_real)  XDIFF,YDIFF,ZDIFF,RIJ2,term1,term2,K
    real(chm_real)  X(*), Y(*), Z(*)

    !     make sure units are consistent
    !       print *,"FQENER eps_gb",eps_gb
    !       print "(f10.5)",alph_gb(1:natom)
    !     prefactor with solute and solvent dielectrics
    K = -CCELEC* ( (1.0/EPS) - (1.0/Eps_GB) )

    !     looping structure not the most efficient for now
    DO I = 1, NATOM
       DO J = 1, NATOM
          IF (I /= J) THEN
             XDIFF = X(I) - X(J)
             YDIFF = Y(I) - Y(J)
             ZDIFF = Z(I) - Z(J)
             RIJ2 = XDIFF**2 + YDIFF**2 + ZDIFF**2
             term1 = alph_gb(I)*alph_gb(J)
             !     THE 8.0  is for P6 = 8.0 in GBMV methods
             term2 = RIJ2/(8.0*term1)  
             DCH(I) = DCH(I) + ( -1.0*0.5 * K * CG(J)/ &
                  (sqrt( RIJ2 + term1*EXP(-term2) ) ) )
          ELSE
             DCH(I) = DCH(I) + ( -1.0*0.5*K*( CG(I)/alph_gb(I) )  )
          ENDIF
       ENDDO !  J LOOP
    ENDDO    !  I LOOP
    RETURN
  END  SUBROUTINE  GBFONCHEQ

  SUBROUTINE CHECKETA(QCHEQRDPRM)
    !-----------------------------------------------------------------------
    !     Routine to check to see the state of hardness parameters
    !     for CHEQ code.
    !     IF indeed one wishes to use CHEQ, then they have to have
    !     read in the correct CHEQ model paramater.s
    !     IF the parameters haven't been read, but the user tries to
    !     run any CHEQ method, there should be an alert and die.

    use number
    use psf
    use parallel

    INTEGER I
    LOGICAL FOUND,QCHEQRDPRM

    QCHEQRDPRM = .TRUE.

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 

       FOUND = .FALSE.
       DO I = 1, NATOM
          IF (MOLTYPE(I) /= 9) THEN
             IF (EHA(I) <= 0.0) FOUND = .TRUE.
          ENDIF
       ENDDO

       IF (FOUND) THEN
          QCHEQRDPRM = .FALSE.
       ELSE
          QCHEQRDPRM = .TRUE.
       ENDIF

#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END  SUBROUTINE CHECKETA

  SUBROUTINE CHECKQNORM(QCHEQNORM)
    !-----------------------------------------------------------------------
    !     Routine to check that there are no CHEQ partitions defined
    !     with zero atoms; this indicates some error and must be corrected
    !     by the user before any valid CHEQ method is used.


    use number
    use psf
    use stream
    use parallel

    LOGICAL FOUND,QCHEQNORM

    QCHEQNORM = .TRUE.

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 

       FOUND = .FALSE.
       IF (QNPART <= 0) FOUND = .TRUE.

       IF (FOUND) THEN
          QCHEQNORM = .FALSE.
       ELSE
          QCHEQNORM = .TRUE.
       ENDIF

       FOUND = .FALSE.

#if KEY_PARALLEL==1
    ENDIF     
#endif
    !
    RETURN
  END  SUBROUTINE CHECKQNORM
#endif /* (cheq_main)*/

  SUBROUTINE CPCGIMG(IMATTRX)
    !----------------------------------------------------------------------
    ! THIS ROUTINE COPIES CG FROM MAIN SET TO IMAGE ATOMS
    ! FOR USE IN CHARGE DYNAMICS
    !
    use image
    use psf
    !
    INTEGER IMATTRX(*)
    !
    INTEGER I,J
    IF (NATIM > NATOM) THEN
       DO I=NATOM+1,NATIM
          J=IMATTRX(I)
          CG(I)=CG(J)
       ENDDO
    ENDIF
    RETURN
  END  SUBROUTINE CPCGIMG

end module cheq

