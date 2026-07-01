module hbanal_mod
  use chm_kinds
  use chm_types
  use dimens_fcm
  implicit none

  !  distance and lifetime histograms in hbond analysis
  !
  !  MXRH   Allocation size for R-histogram
  !  IRHIST Pointer to R-histogram
  !  IUNRH  Unit to write histogram to (don't write if < 1, defaults to OUTU)
  !  QHBHIST True if R-HISTOGRAM is to be accumulated
  !  DRH    Spacing in R-histogram
  !  RHMAX  Distance corrsponding to last bin
  !      and same for T-histograms for variables with T instead of R in the name
  !
  integer,allocatable,dimension(:) :: IRHIST, ITHIST
  INTEGER MXRH,MXTH,IUNRH,IUNTH
  REAL(chm_real) DRH,DTH,RHMAX,THMAX
  LOGICAL QHBHIST
#if KEY_PARALLEL==1
  logical pario_q ! do we process this in parallel
#endif

contains

  subroutine hbanal_init
    qhbhist=.false.
    return
  end subroutine hbanal_init

  SUBROUTINE HBANAL(COMLYN,COMLEN,ISLCT,JSLCT,MODE)
    !-----------------------------------------------------------------------
    !     This routine analyses trajectories for hydrogen bonds between
    !     atoms specified in ISLCT and JSLCT.
    !
    !     L. Nilsson, January 1994, KI/CSB
    !     Modified to allow all hb-pairs to be analyzed from trajectory. L. Nilsson, May 2000
    !     Modified to allow PBC (using pbound code) cubic,TO and RHDO.  L. Nilsson, March 2005,
    !     with input from G. Lamoureux, Cornell Med School.
    !
    use exfunc
    use consta
    use number
    use bases_fcm
    use coord
    use inbnd
    use psf
    use stream
    use string
    use memory
#if KEY_PBOUND==1
    use pbound               
#endif
    use cvio,only:trjspc
    use chutil,only:atomid
    !---   use nbutil_module,only:gtnbct
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN, ISLCT(NATOM),JSLCT(NATOM)
    integer,allocatable,dimension(:) :: HEAVY
    LOGICAL,allocatable,dimension(:) :: AFLAG,DFLAG
    CHARACTER(len=4) MODE
    !
    INTEGER, PARAMETER :: MAXDA=500,WDMAX=4
    INTEGER IDN,IHBAC,JDN,JHBAC,ISEL,JSEL,NBDON,NBAC
    INTEGER IDA,JDALST,HBNO,HBLIFE,HBLST,HBCNT,HBRDG,BDALST
    INTEGER IUNIT,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP,HBAGE
    INTEGER TEMP,FREEAT
    real(chm_real) CUT,CUTA,TCUT,PCUT
    CHARACTER(len=WDMAX) BRIDGE
    CHARACTER(len=8) SID,RID,REN,IUPAC
    LOGICAL QBRIDGE,QVERB,QVDWR
    INTEGER I,J,WDLEN,MXJDA,MXBDA


    call chmalloc('hbanal.src','HBANAL','AFLAG',NATOM,log=AFLAG)
    call chmalloc('hbanal.src','HBANAL','DFLAG',NATOM,log=DFLAG)
    call chmalloc('hbanal.src','HBANAL','HEAVY',NATOM,intg=HEAVY)
    !
    ! Parse rest of commandline
    ! Default H--X 2.4A (reasonable criterion, DeLoof et al (1992) JACS 114,4028)
    ! Allow variants of keywords: CUT or CUTHB for distance,
    ! CUTA, CUTHA, or CUTHBA for angle
    ! For the CONTact mode no angle cutoff should be allowed
    CUTA=GTRMF(COMLYN,COMLEN,'CUTHBA',NINE99)
    CUTA=GTRMF(COMLYN,COMLEN,'CUTHA',CUTA)
    CUTA=GTRMF(COMLYN,COMLEN,'CUTA',CUTA)

    QVDWR=(INDXRA(COMLYN,COMLEN,'VDWR',4,.TRUE.)  >  0)
    IF(QVDWR)THEN
      CUT=GTRMF(COMLYN,COMLEN,'CUTHB',-(ONE+PTONE))
    ELSE
      CUT=GTRMF(COMLYN,COMLEN,'CUTHB',TWOPT4)
    ENDIF
    CUT=GTRMF(COMLYN,COMLEN,'CUT',CUT)
    IF(MODE == 'CONT') CUTA=NINE99

    ! cutvalue for life times to be considered interesting
    TCUT=GTRMF(COMLYN,COMLEN,'TCUT',ZERO)
    ! cutoff for fraction of time as hbond to be considered interesting
    PCUT=GTRMF(COMLYN,COMLEN,'OCUT',ZERO)
    ! amount of output
    QVERB=(INDXRA(COMLYN,COMLEN,'VERB',4,.TRUE.)  >  0)
    ! REQUIRE that a FIRStu be specified if the analysis is to be performed
    ! on a trajectory, so we can take missing FIRStu to mean use current coord
#if KEY_PARALLEL==1
    pario_q = (INDXRA(COMLYN,COMLEN,'PLBL',4,.TRUE.)  >  0)
#endif
    IF(INDX(COMLYN,COMLEN,' FIRS',5)  >  0)THEN
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)
    ELSE
       NUNIT= -1
    ENDIF
    IUNIT=GTRMI(COMLYN,COMLEN,'IUNI',OUTU)
    QBRIDGE=INDEX(COMLYN,'BRID') > 0
    IF(QBRIDGE) CALL GTRMWA(COMLYN,COMLEN,'BRID',4,BRIDGE,WDMAX,WDLEN)
    !
    DRH=GTRMF(COMLYN,COMLEN,'DRH',PT05)
    RHMAX=GTRMF(COMLYN,COMLEN,'RHMA',TEN)
    DTH=GTRMF(COMLYN,COMLEN,'DTH',FIVE)
    THMAX=GTRMF(COMLYN,COMLEN,'THMA',THOSND)
    IUNRH=GTRMI(COMLYN,COMLEN,'IRHI',-1)
    IUNTH=GTRMI(COMLYN,COMLEN,'ITHI',-1)
    MXRH=RHMAX/DRH + 1
    MXTH=THMAX/DTH + 1

    call chmalloc('hbanal.src','HBANAL','IRHIST',MXRH+1,intg=IRHIST)
    call chmalloc('hbanal.src','HBANAL','ITHIST',MXTH+1,intg=ITHIST)

    QHBHIST=IUNRH > 0
    irhist(1:mxrh+1)=0
    ithist(1:mxth+1)=0

    IF(INDXA(COMLYN,COMLEN,'PBC') > 0)THEN
       ! extract PBC info (parsing is similar to SUBROUTINE BOUND)
#if KEY_PBOUND==1
       QBOUN=.TRUE.
       QCUBOUN=INDXA(COMLYN,COMLEN,'CUBIC') > 0
       QTOBOUN=INDXA(COMLYN,COMLEN,'TO') > 0
       QRHBOUN=INDXA(COMLYN,COMLEN,'RHDO') > 0
       XSIZE=GTRMF(COMLYN,COMLEN,'BOXL',ZERO)
       XSIZE=GTRMF(COMLYN,COMLEN,'XSIZE',XSIZE)
       YSIZE=GTRMF(COMLYN,COMLEN,'YSIZE',XSIZE)
       ZSIZE=GTRMF(COMLYN,COMLEN,'ZSIZE',XSIZE)
       IF(XSIZE < RSMALL) CALL WRNDIE(-2,'<HBANAL>', &
            'Boxsize has to be specified with PBC')
       BOXINV=ONE/XSIZE
       BOYINV=ONE/YSIZE
       BOZINV=ONE/ZSIZE
    ELSE   
       QBOUN=.FALSE.
#else /**/
       CALL WRNDIE(-2,'<HBANAL>', &
            'PBOUND code not compiled. Required for PBC')
#endif 
    ENDIF
    !
    ! Make sure non-bond exclusions are set up
    COMLEN=0
    CALL GTNBCT(COMLYN,COMLEN,BNBND) 
    ! Should be changed so that we don't need coordinates present
    ! Comlyn should be completely parsed, so use it to make update as painless
    ! as possible
    ! try this:
    !      IF(NNB14  <=  0) THEN
    ! Try to get the exclusion list setup with minimal fuzz:
    !         COMLYN='CUTNB 2.0 '
    !         COMLEN=10
    !         CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,.TRUE.,
    !     &            .FALSE.,.TRUE.,.FALSE.,.FALSE.,0,0,0,0,0,0,0)
    !      ENDIF
    !
    ! The acceptor/donor lists in the PSF are not sorted, and we can make
    ! better use of a lookuptable
    DO I=1,NATOM
       AFLAG(I)=.FALSE.
       DFLAG(I)=.FALSE.
    ENDDO
    IF(MODE == 'HBON')THEN
       DO I=1,NACC
          AFLAG(IACC(I))=.TRUE.
       ENDDO
       DO I=1,NDON
          DFLAG(IHD1(I))=.TRUE.
          ! HEAVY points to the heavy atom (donor) to which the H is covalently
          ! attached
          HEAVY(IHD1(I))=IDON(I)
       ENDDO
    ELSEIF(MODE == 'CONT')THEN
       !
       ! Now let us simply denote all selected atoms in the I and J sets
       ! as donors (I) and accpetors (J) (if QBRIDGE, the J atoms will
       ! also be faked as donors, with the bridging atoms designed acceptors)
       !
       DO I=1,NATOM
          IF(ISLCT(I) > 0)THEN
             DFLAG(I)=.TRUE.
             ! make sure HEAVY has a valid pointer, even though it should not be
             ! used!
             HEAVY(I)=I
          ENDIF
          IF(JSLCT(I) > 0)THEN
             IF(QBRIDGE) THEN
                DFLAG(I)=.TRUE.
                HEAVY(I)=I
             ELSE
                AFLAG(I)=.TRUE.
             ENDIF
          ENDIF
       ENDDO
    ELSE
       !
       ! it should not be possible to get here...
       CALL WRNDIE(-2,'<HBANAL>','Unknown mode: '//MODE)
       call hbanal_dealloc
       RETURN
    ENDIF
    !
    !
    ! Find out how many possible bridging groups there would be
    IF(QBRIDGE)THEN
       NBDON=0
       NBAC=0
       DO I=1,NATOM
          CALL ATOMID(I,SID,RID,REN,IUPAC)
          IF(REN  ==  BRIDGE)THEN
             IF(MODE == 'HBON')THEN
                IF(DFLAG(I))THEN
                   NBDON=NBDON+1
                ELSE IF(AFLAG(I))THEN
                   NBAC=NBAC+1
                ENDIF
             ELSE
                !
                ! just fake all bridge atoms as acceptors AND donors in this case
                !  possilby problematic with overlapping selections?
                NBAC=NBAC+1
                NBDON=NBDON+1
                DFLAG(I)=.TRUE.
                AFLAG(I)=.TRUE.
             ENDIF
          ENDIF
       ENDDO
    ENDIF

    ! Compute sizes and remove all non donor/acceptors;
    ! IN case of contact analysis everything may be flagged as
    ! both donor and acceptor..
    ISEL=0
    JSEL=0
    IDN=0
    IHBAC=0
    JDN=0
    JHBAC=0
    DO I=1,NATOM
       IF(ISLCT(I) > 0)THEN
          IF(DFLAG(I).OR.AFLAG(I))THEN
             ISEL=ISEL+1
             IF(DFLAG(I))IDN=IDN+1
             IF(AFLAG(I)) IHBAC=IHBAC+1
          ELSE
             ISLCT(I)=0
          ENDIF
       ENDIF
       IF(JSLCT(I) > 0)THEN
          IF(DFLAG(I).OR.AFLAG(I))THEN
             JSEL=JSEL+1
             IF(DFLAG(I)) JDN=JDN+1
             IF(AFLAG(I)) JHBAC=JHBAC+1
          ELSE
             JSLCT(I)=0
          ENDIF
       ENDIF
    ENDDO
    !
    IF(PRNLEV  >=  2)THEN
       IF(MODE == 'HBON')THEN
          WRITE(OUTU,900) 'hydrogen bond',IDN,IHBAC,JDN,JHBAC
       ELSE
          WRITE(OUTU,900) 'close contact',IDN,IHBAC,JDN,JHBAC
       ENDIF
900    FORMAT(//' Analysing ',A,' patterns for:', &
            /'              #donors  #acceptors', &
            /' 1st sel:',2I12, &
            /' 2nd sel:',2I12,/)
       IF(QBRIDGE)THEN
          WRITE(OUTU,910) NBDON,NBAC
910       FORMAT(' Bridges:',2I12,/)
       ENDIF
#if KEY_PBOUND==1
       IF(QBOUN)THEN
          IF(QCUBOUN) WRITE(OUTU,915) XSIZE,YSIZE,ZSIZE 
          IF(QTOBOUN) WRITE(OUTU,916) XSIZE
          IF(QRHBOUN) WRITE(OUTU,917) XSIZE
915       FORMAT(/'Using cubic PBC with sides:',3F10.2/)
916       FORMAT(/'Using truncated octahedron  PBC, A=',F10.2/)
917       FORMAT(/'Using rhombic dodecahedron  PBC, A=',F10.2/)
       ENDIF
#endif 
    ENDIF
    !
    IF(QBRIDGE)THEN
       IF(NBDON+NBAC  ==  0)THEN
          CALL WRNDIE(1,'<HBANAL>','NO BRIDGE ACCEPTOR/DONORS FOUND')
          call hbanal_dealloc
          RETURN
       ENDIF
       IF((IDN+IHBAC  == 0) .OR.(JDN+JHBAC  ==  0))THEN
          CALL WRNDIE(1,'<HBANAL>','NO ACCEPTOR/DONORS TO BRIDGE TO')
          call hbanal_dealloc
          RETURN
       ENDIF
    ELSE
       IF(IDN*JHBAC+IHBAC*JDN  ==  0)THEN
          CALL WRNDIE(1,'<HBANAL>','NO DONOR-ACCEPTOR PAIRS FOUND')
          call hbanal_dealloc
          RETURN
       ENDIF
    ENDIF
    MXJDA=MAX(JDN,JHBAC)

    IF(QBRIDGE)THEN
       MXBDA=MAX(NBDON,NBAC)
    ELSE
       ! to allow array bounds checking w/o being trapped on the call...
       MXBDA=1
    ENDIF
    !
    !     check our non-bond exclusion list
    IF(NNB14  <=  0) &
         CALL WRNDIE(-1,'<HBANAL>','NO EXCLUSIONS')

    CALL HBAN1(MODE,ISEL,ISLCT,JSEL,JSLCT,JDN,JHBAC,CUT,CUTA, &
         TCUT,PCUT,QVERB,HEAVY,IUNIT,NUNIT, &
         FIRSTU,NBEGIN,NSKIP,NSTOP,MAXDA,MXJDA,MXBDA, &
         QVDWR,QBRIDGE,BRIDGE,NBDON,NBAC, &
         AFLAG,DFLAG, &
         BNBND%IBLO14,BNBND%INB14,ITHIST)
    !
    ! Free space
    call chmdealloc('hbanal.src','HBANAL','IRHIST',MXRH,intg=IRHIST)
    call chmdealloc('hbanal.src','HBANAL','ITHIST',MXTH,intg=ITHIST)

    ! Set flag telling that the space above is NOT available
    QHBHIST=.FALSE.
#if KEY_PBOUND==1
    IF(PRNLEV >= 2 .AND. QBOUN) WRITE(OUTU,'(/A/)')  &
         'Hardwired MI PBC turned OFF!'
    QBOUN=.FALSE.
#endif 
    call hbanal_dealloc
    RETURN
  contains

    subroutine hbanal_dealloc
      call chmdealloc('hbanal.src','hbanal','AFLAG',NATOM,log=AFLAG)
      call chmdealloc('hbanal.src','hbanal','DFLAG',NATOM,log=DFLAG)
      call chmdealloc('hbanal.src','hbanal','HEAVY',NATOM,intg=HEAVY)
      return
    end subroutine hbanal_dealloc

  END SUBROUTINE HBANAL

  SUBROUTINE HBAN1(MODE,ISEL,ISLCT,JSEL,JSLCT,JDN,JHBAC,CUT, &
       CUTA,TCUT,PCUT,QVERB,HEAVY,IUNIT, &
       NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP,MAXDA,MXJDA,MXBDA, &
       QVDWR,QBRIDGE,BRIDGE,NBDON,NBAC, &
       AFLAG,DFLAG,IBLO14,INB14,THIST)
    !-----------------------------------------------------------------------
    use exfunc
    use chutil
    use number
    use coord
    use consta
    use ctitla
    use cvio
    use psf
#if KEY_PARALLEL==1
    use parallel
    use mpi
#endif
    use stream
    use memory
#if KEY_PBOUND==1
    use pbound
    use image
#endif 
    use param_store, only: set_param
    use select, only: filsky

    implicit none

    ! For PBC handling, L. Nilsson
#if KEY_PBOUND==1
    LOGICAL QXTL
#endif 
    INTEGER ISEL,ISLCT(NATOM),JSEL,JSLCT(NATOM),HEAVY(NATOM)
    INTEGER IUNIT,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP,MAXDA
    INTEGER JDN,JHBAC,NBDON,NBAC,MXJDA,MXBDA
    LOGICAL AFLAG(NATOM),DFLAG(NATOM)
    ! CHECK ON OPTIMAL ORDER FOR HBLST INDICES!!!!!!!!
    INTEGER IBLO14(*),INB14(*)
    INTEGER THIST(0:MXTH)
    real(chm_real)  CUT,CUTA,TCUT,PCUT
    CHARACTER(len=4) MODE
    CHARACTER(len=*) BRIDGE
    LOGICAL QBRIDGE,QVERB,QVDWR
    !
    integer,allocatable,target,dimension(:) :: ispace
    integer,pointer,dimension(:) :: ida,jda,hbno,hblife,hbcnt,freeat
    integer,allocatable,dimension(:,:) :: jdalst,hblst,hbrdg,hbage,bdalst
    real(chm_real4),allocatable,dimension(:) :: temp
    integer :: bdalstsz1,bdalstsz2,hbrdgsz1,hBrdgsz2,lhblist_siz2, &
         jdasz

    INTEGER I,II,J,JJ,K,KK,L,LL,UNIT,NSTEP,ISTEP,ISTATS,IFILE,J2
    INTEGER NFREAT,NDEGF,RNSAVV,NPAIR(0:1),NBRDG(0:1),NBDA(0:1)
    INTEGER M,MM,JRES,IS,IQ,IATOM,NBA,NBD,IA,ID,IEV
    INTEGER MINSTEP,IT1
    INTEGER, PARAMETER :: MAXBAT=20
    INTEGER OFFS(0:1,MAXBAT)
    LOGICAL DONE,QCLOSE, Q2CLOSE,LUSED
    CHARACTER(len=8) SID,RID,REN,IUPAC,SID2,RID2,REN2,IUPAC2
    CHARACTER(len=8) SID3,RID3,REN3,IUPAC3
    real(chm_real) XI,YI,ZI,S,AVNO,AVLIFE,STPTIME,DELTA,XSTEP
    real(chm_real) XJ,YJ,ZJ, TSUM,AVSUM,PS,HBDIST,HBANG, &
         AVINO,AVILIF,AVLIFE1
    CHARACTER(len=4) ::COORHD='CORD',VELHD='VELD'
    CHARACTER(len=13) LEADTXT
    INTEGER, PARAMETER :: DONOR=0,ACCEPTOR=1

    integer,allocatable,dimension(:,:,:) :: lhblist
    integer,allocatable,dimension(:) :: jsel2,hbdefi,hbdefj,hbdefb
    integer lhblist_siz1,lhbsiz_siz2,jsel2_siz
    !
#if KEY_PARALLEL==1
    ! for parallel I/O, assume idleng=8 for format 901
    integer, parameter :: pario_len = 10*8+38
    character(len=pario_len),allocatable,dimension(:) :: pario_lines
    integer pario_count ! how many lines need to be send/received
    integer pario_i, pario_tag, pario_stat(mpi_status_size), pario_err
    integer pario_st, pario_sk
    integer,allocatable,dimension(:) :: pario_count_all
#endif
    ! used in both parallel/OpenMP & [NON-OpenMP and/or NON-parallel]
    integer pario_npair
    logical pario_qexcl
    logical, allocatable, dimension(:) :: pario_qclose
    !
    IF(QVERB)THEN
       LHBLIST_siz1=jsel
       LHBLIST_siz2=isel
       JSEL2_siz=natom
       call chmalloc('hbanal.src','HBAN1','HBDEFI',natom,intg=hbdefi)
       call chmalloc('hbanal.src','HBAN1','HBDEFJ',natom,intg=hbdefj)
       hbdefi=0
       hbdefj=0 
       IF(QBRIDGE) THEN
         call chmalloc('hbanal.src','HBAN1','HBDEFB',natom,intg=hbdefb)
         hbdefb=0
       ENDIF
    ELSE
       LHBLIST_siz1=1
       LHBLIST_siz2=1
       JSEL2_siz=1
    ENDIF
#if KEY_PARALLEL==1
    if(pario_q)then
       allocate(pario_lines(maxda),pario_count_all(numnod))
    endif
#endif
    call chmalloc('hbanal.src','HBAN1','LHBLIST', &
         2,lhblist_siz1,lhblist_siz2,intg=LHBLIST)
    call chmalloc('hbanal.src','HBAN1','JSEL2',jsel2_siz,intg=JSEL2)
    call chmalloc('hbanal.src','HBAN1','ispace',4*isel+2*natom,intg=ispace)
    ida    => ispace(0*isel+1: 1*isel)
    hbno   => ispace(1*isel+1: 2*isel)
    hblife => ispace(2*isel+1: 3*isel)
    hbcnt  => ispace(3*isel+1: 4*isel)
    freeat => ispace(0*isel+1: 1*isel+natom)
    jda    => ispace(0*isel+natom+1: 1*isel+2*natom)


    call chmalloc('hbanal.src','HBAN1','temp',natom,cr4=temp)
    call chmalloc('hbanal.src','HBAN1','HBLST',MAXDA,ISEL,intg=hblst)
    call chmalloc('hbanal.src','HBAN1','HBAGE',MAXDA,ISEL,intg=hbage)
    call chmalloc('hbanal.src','HBAN1','JDALST',2,mxjda,intg=jdalst,lbou=0)
    if(qbridge)then
       jdasz=natom
       hbrdgsz1=maxda
       hbrdgsz2=isel
       bdalstsz1=2
       bdalstsz2=mxbda
    else
       jdasz=1
       hbrdgsz1=1
       hbrdgsz2=1
       bdalstsz1=1
       bdalstsz2=1
    endif
    call chmalloc('hbanal.src','HBAN1','HBRDG', &
         hbrdgsz1,hbrdgsz2,intg=hbrdg)
    call chmalloc('hbanal.src','HBAN1','BDALST', &
         bdalstsz1,bdalstsz2,intg=bdalst,lbou=0)

    IF(MODE == 'HBON')THEN
       LEADTXT='hydrogen bond'
    ELSE
       LEADTXT='close contact'
    ENDIF
    ! Make indexlists, initialize counters
    CALL MAKIND(NATOM,ISLCT,ISLCT,I)
    IF(I /= ISEL) CALL WRNDIE(-2,'<HBANAL>','I-INDEX PROBLEM')
    CALL MAKIND(NATOM,JSLCT,JSLCT,I)
    IF(I /= JSEL) CALL WRNDIE(-2,'<HBANAL>','J-INDEX PROBLEM')
    !
    !
    NPAIR(DONOR)=JDN
    NPAIR(ACCEPTOR)=JHBAC
    DO I=1,ISEL
       IF(DFLAG(ISLCT(I)))THEN
          IDA(I)=DONOR
       ELSE IF(AFLAG(ISLCT(I)))THEN
          IDA(I)=ACCEPTOR
       ENDIF
    ENDDO
    JJ=0
    KK=0
    !
    ! Find all acceptors,donors in J-selection
    DO J=1,JSEL
       IF(QVERB) JSEL2(JSLCT(J))=J
       IF(AFLAG(JSLCT(J)))THEN
          JJ=JJ+1
          JDALST(ACCEPTOR,JJ)=JSLCT(J)
          IF(QBRIDGE) JDA(JSLCT(J))=ACCEPTOR
       ENDIF
       IF(DFLAG(JSLCT(J)))THEN
          KK=KK+1
          JDALST(DONOR,KK)=JSLCT(J)
          IF(QBRIDGE) JDA(JSLCT(J))=DONOR
       ENDIF
    ENDDO
    ! Check that everything sums up OK...
    IF(JJ  /=  NPAIR(ACCEPTOR))THEN
       IF(PRNLEV  >=  5)  THEN
          WRITE(OUTU,*) 'HBANAL> # Acceptor pairs',JJ,NPAIR(ACCEPTOR)
       ENDIF
       CALL WRNDIE(-3,'<HBANAL>', &
            '2nd selection, acceptor pair problem')
    ENDIF
    IF(KK  /=  NPAIR(DONOR))THEN
       IF(PRNLEV  >=  5)  THEN
          WRITE(OUTU,*) 'HBANAL> # Donor pairs',KK,NPAIR(DONOR)
       ENDIF
       CALL WRNDIE(-3,'<HBANAL>', &
            '2nd selection, donor pair problem')
    ENDIF

    ! List what we have, if PRNLEV is high.
    IF(PRNLEV  >  5)THEN
       WRITE(OUTU,*) ' HBANAL>  First selection.'
       DO I=1,ISEL
          II=ISLCT(I)
          CALL ATOMID(II,SID,RID,REN,IUPAC)
          WRITE(OUTU,915) II,SID(1:idleng),REN(1:idleng), &
               RID(1:idleng),IUPAC(1:idleng),IDA(I),HEAVY(II)
915       FORMAT(I8,4(1X,A),2I6)
       ENDDO
       WRITE(OUTU,*) ' '
       WRITE(OUTU,*) ' HBANAL>  Second selection. ACCEPTORS.'
       DO J=1,NPAIR(ACCEPTOR)
          JJ=JDALST(ACCEPTOR,J)
          CALL ATOMID(JJ,SID,RID,REN,IUPAC)
          WRITE(OUTU,915) JJ,SID(1:idleng),REN(1:idleng), &
               RID(1:idleng),IUPAC(1:idleng)
       ENDDO
       WRITE(OUTU,*) ' '
       WRITE(OUTU,*) ' HBANAL>  Second selection. DONORS.'
       DO J=1,NPAIR(DONOR)
          JJ=JDALST(DONOR,J)
          CALL ATOMID(JJ,SID,RID,REN,IUPAC)
          WRITE(OUTU,915) JJ,SID(1:idleng),REN(1:idleng), &
               RID(1:idleng),IUPAC(1:idleng),HEAVY(JJ)
       ENDDO
       WRITE(OUTU,*) ' '
    ENDIF
    IF(QBRIDGE)THEN
       ! Find all acceptors,donors in possible bridging residues
       NBRDG(ACCEPTOR)=NBAC
       NBRDG(DONOR)=NBDON
       JJ=0
       KK=0
       L=0
       DO J=1,NATOM
          CALL ATOMID(J,SID,RID,REN,IUPAC)
          IF(REN  ==  BRIDGE)THEN
             ! Now is the time to learn about our bridging residue
             IF(L  ==  0)THEN
                L=1
                LL=IBASE(GETRES(J,IBASE,NRES))+1
                NBA=0
                NBD=0
                DO I=LL,IBASE(GETRES(J,IBASE,NRES)+1)
                   IF(AFLAG(I))THEN
                      NBA=NBA+1
                      OFFS(ACCEPTOR,NBA)=I-LL
                   ELSEIF(DFLAG(I))THEN
                      NBD=NBD+1
                      OFFS(DONOR,NBD)=I-LL
                   ENDIF
                ENDDO
                NBDA(ACCEPTOR)=NBA
                NBDA(DONOR)=NBD
             ENDIF
             IF(AFLAG(J))THEN
                JJ=JJ+1
                BDALST(ACCEPTOR,JJ)=J
             ELSE IF(DFLAG(J))THEN
                KK=KK+1
                BDALST(DONOR,KK)=J
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV  >  5 )THEN
          WRITE(OUTU,*) 'Bridge donors:' ,Nbrdg(donor)
          WRITE(OUTU,'(10I7)') (bdalst(donor,j),J=1,nbrdg(donor))
          WRITE(OUTU,*) 'Bridge acceptors:', Nbrdg(acceptor)
          WRITE(OUTU,'(10I7)') &
               (bdalst(acceptor,j),J=1,nbrdg(acceptor))
          WRITE(OUTU,*) 'Donor offsets:', NBD
          WRITE(OUTU,'(10I7)') (offs(donor,j),j=1,nbd)
          WRITE(OUTU,*) 'Acceptor offsets:', NBA
          WRITE(OUTU,'(10I7)') (offs(acceptor,j),j=1,nbA)
          WRITE(OUTU,*)
       ENDIF
    ENDIF
    !
    ! Initialize counters
    DO I=1,ISEL
       HBNO(I)=0
       HBLIFE(I)=0
       HBCNT(I)=0
       DO J=1,MAXDA
          HBLST(J,I)=0
          HBAGE(J,I)=0
       ENDDO
       IF(QBRIDGE)THEN
          DO J=1,MAXDA
             HBRDG(J,I)=0
          ENDDO
       ENDIF
    ENDDO
    LHBLIST=0
    NSTEP=0
    UNIT=FIRSTU
    ISTATS=1
    DONE=.FALSE.
    !
    !
100 CONTINUE
    ! Trajectory or current coordinates?
    IF(NUNIT  <=  0)THEN
       DONE=.TRUE.
       NSTEP=1
       STPTIME=ZERO
    ELSE
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
            TEMP, &
            NATOM,FREEAT,NFREAT, &
            FIRSTU,NUNIT,UNIT,IFILE, &
            ISTEP,ISTATS,NDEGF,DELTA, &
            NBEGIN,NSTOP,NSKIP,RNSAVV,COORHD,VELHD, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       NSTEP=NSTEP+1
       IF(NSTEP  ==  1)THEN
          STPTIME=NSKIP*DELTA*TIMFAC
       ENDIF
       IF(ISTATS ==  -1) DONE=.TRUE.
    ENDIF
#if KEY_PBOUND==1
    ! Get appropriate sizes from trajectory/crystal data structure if available.
    ! Assume XTLABC=ZERO means that it should not be used
    IF(QBOUN)THEN
       QXTL=.FALSE.
       DO I=1,6
          IF(XTLABC(I) /= ZERO) QXTL=.TRUE. 
       ENDDO
       IF(QXTL) THEN
          CALL XTLLAT(XUCELL,XTLABC) 
          XSIZE=XUCELL(1)
          YSIZE=XUCELL(2)
          ZSIZE=XUCELL(3)
          BOXINV=ONE/XSIZE
          BOYINV=ONE/YSIZE
          BOZINV=ONE/ZSIZE
       ENDIF
    ENDIF
#endif 
    IF(QVERB.AND.NUNIT > 0.AND.NSTEP == 1)THEN
       WRITE(IUNIT,888)
888    FORMAT(/'I-atom',17X,'J-atom',17X,'(Bridge)', &
            7X,'Lifetime Endtime')
    ENDIF

#if KEY_PARALLEL==1
    if(.not.pario_q) then
       pario_st=1
       pario_sk=1
    else
       pario_st=mynodp
       pario_sk=numnod
    endif
    loop1: DO I=pario_st,ISEL,pario_sk
#else
      loop1: DO I=1,ISEL
#endif
       IATOM=ISLCT(I)
       !
       ! Go through list of existing HBonds, and take care of the broken ones first
       ! (NOTE THAT THIS IS WHERE THE LIFE TIME IS ACCUMULATED
       !  so when the last coordinate set is read a last sum up has to be done)
#if KEY_PARALLEL==1
       pario_count=0
#endif

       j_loop: DO J=1,MAXDA
          in_list: IF(HBLST(J,I)  >  0)THEN
             JJ=HBLST(J,I)
             bridge_block: IF(QBRIDGE)THEN
                ! Are we still close enough to this bridging molecule?
                ! ie, is there at least one possible partner within reach?
                QCLOSE=.FALSE.
                DO L=1,NBDA(1-IDA(I))
                   KK=HBRDG(J,I)+OFFS(1-IDA(I),L)
                   QCLOSE=QDISANG(IATOM,KK,CUT,CUTA,DFLAG(IATOM), &
                        HEAVY,QVDWR) .OR. QCLOSE
                ENDDO
                ! If so, is this still close to the other end of the selection?
                IF(QCLOSE)THEN
                   Q2CLOSE=.FALSE.
                   DO L=1,NBDA(1-JDA(JJ))
                      KK=HBRDG(J,I)+OFFS(1-JDA(JJ),L)
                      Q2CLOSE= Q2CLOSE .OR. &
                           QDISANG(JJ,KK,CUT,CUTA,DFLAG(JJ),HEAVY,QVDWR)
                   ENDDO
                   QCLOSE=Q2CLOSE
                ENDIF
                IF(.NOT.QCLOSE)THEN
                   PS=HBAGE(J,I)*STPTIME
                   ! Lifetime histogram
                   IT1=MIN(MXTH,INT(PS/DTH))
                   THIST(IT1)=THIST(IT1)+1
                   IF(PS > TCUT)THEN
                      HBLIFE(I)=HBLIFE(I)+HBAGE(J,I)
                      HBCNT(I)=HBCNT(I)+1
                      IF(QVERB)THEN
                         CALL ATOMID(IATOM,SID,RID,REN,IUPAC)
                         CALL ATOMID(JJ,SID2,RID2,REN2,IUPAC2)
                         CALL ATOMID(KK,SID3,RID3,REN3,IUPAC3)
#if KEY_PARALLEL==1
                         if(mynod==0) then
#endif
                            WRITE(IUNIT,901) SID(1:idleng),RID(1:idleng), &
                                 REN(1:idleng),IUPAC(1:idleng), &
                                 SID2(1:idleng),RID2(1:idleng), &
                                 REN2(1:idleng),IUPAC2(1:idleng), &
                                 SID3(1:idleng),RID3(1:idleng), &
                                 PS,NSTEP*STPTIME
#if KEY_PARALLEL==1
                         else if (pario_q) then
                            pario_count = pario_count +1
                            WRITE(pario_lines(pario_count),901) &
                                 SID(1:idleng),RID(1:idleng), &
                                 REN(1:idleng),IUPAC(1:idleng), &
                                 SID2(1:idleng),RID2(1:idleng), &
                                 REN2(1:idleng),IUPAC2(1:idleng), &
                                 SID3(1:idleng),RID3(1:idleng), &
                                 PS,NSTEP*STPTIME
                         endif
#endif
                         J2=JSEL2(JJ)
                         LHBLIST(1,J2,I)=LHBLIST(1,J2,I)+HBAGE(J,I)
                         LHBLIST(2,J2,I)=LHBLIST(2,J2,I)+1
                      ENDIF
                   ENDIF
                   HBLST(J,I)=0
                   HBAGE(J,I)=0
                   HBRDG(J,I)=0
                ENDIF
901             FORMAT(4(1X,A),' - ',4(1X,A),5X,2(1X,A),2F10.2)
             ELSE bridge_block
                QCLOSE=QDISANG(IATOM,JJ,CUT,CUTA,DFLAG(IATOM),HEAVY,QVDWR)
                IF(.NOT.QCLOSE)THEN
                   PS=HBAGE(J,I)*STPTIME
                   ! Lifetime histogram
                   IT1=MIN(MXTH,INT(PS/DTH))
                   THIST(IT1)=THIST(IT1)+1
                   IF(PS > TCUT)THEN
                      HBLIFE(I)=HBLIFE(I)+HBAGE(J,I)
                      HBCNT(I)=HBCNT(I)+1
                      IF(QVERB)THEN
                         CALL ATOMID(IATOM,SID,RID,REN,IUPAC)
                         CALL ATOMID(JJ,SID2,RID2,REN2,IUPAC2)
#if KEY_PARALLEL==1
                         if(mynod==0) then
#endif
                            WRITE(IUNIT,901) SID(1:idleng),RID(1:idleng), &
                                 REN(1:idleng),IUPAC(1:idleng), &
                                 SID2(1:idleng),RID2(1:idleng), &
                                 REN2(1:idleng),IUPAC2(1:idleng),' ',' ', &
                                 PS,NSTEP*STPTIME
#if KEY_PARALLEL==1
                         else if (pario_q) then
                            pario_count = pario_count +1
                            WRITE(pario_lines(pario_count),901) &
                                 SID(1:idleng),RID(1:idleng), &
                                 REN(1:idleng),IUPAC(1:idleng), &
                                 SID2(1:idleng),RID2(1:idleng), &
                                 REN2(1:idleng),IUPAC2(1:idleng),' ',' ', &
                                 PS,NSTEP*STPTIME
                         endif
#endif
                         J2=JSEL2(JJ)
                         LHBLIST(1,J2,I)=LHBLIST(1,J2,I)+HBAGE(J,I)
                         LHBLIST(2,J2,I)=LHBLIST(2,J2,I)+1
                      ENDIF
                   ENDIF
                   HBLST(J,I)=0
                   HBAGE(J,I)=0
                ENDIF
             ENDIF bridge_block
          ENDIF in_list
       ENDDO j_loop
    
#if KEY_PARALLEL==1
       !
       ! for parallel communicate the written lines for each atom
       ! with additional bookkeeping this block can be moved after the loop1
       ! to prevent blocking first communicate pario_count-s
       if(pario_q) then
          pario_tag=99
          if (mynod==0) then
             do pario_i=1,numnod-1
                call mpi_recv(pario_count_all(pario_i),1,mpi_integer, &
                     pario_i,pario_tag, mpi_comm_world, pario_stat, pario_err)
             end do
          else
             call mpi_send(pario_count,1,mpi_integer,0, &
                  pario_tag,mpi_comm_world,pario_err)
          endif
          ! now process the output lines
          if (mynod==0) then
             do pario_i=1,numnod-1
                if (pario_count_all(pario_i) > 0) then
                   call mpi_recv(pario_lines,pario_len*pario_count_all(pario_i), &
                        mpi_character, pario_i, pario_tag, mpi_comm_world, &
                        pario_stat, pario_err)
                   ! this could be TRIMmed but then we probably need a loop here?
                   write(outu,'(a)')pario_lines(1:pario_count_all(pario_i))(1:pario_len)
                endif
             end do
          else
             if(pario_count >0) call mpi_send(pario_lines,pario_len*pario_count, &
                  mpi_character,0,pario_tag,mpi_comm_world,pario_err)
          endif
       endif
#endif
       !
       ! Now check all possibilities, for new and prevailing HBonds
       ! (if IDA(I)=ACCEPTOR, THEN 1-IDA(I)=DONOR, which we want to have)
       !
       book_keep1: IF(.NOT.QBRIDGE)THEN

          pario_npair = npair(1-IDA(I))
#if _OPENMP          
          allocate(pario_qclose(pario_npair))
          pario_qclose(1:pario_npair) = .False.
          
          !$OMP PARALLEL PRIVATE(J,JJ,QCLOSE,pario_qexcl)
          !$OMP DO SCHEDULE(static) ordered
#endif
          no_bridge: DO J=1,pario_NPAIR
             JJ=JDALST(1-IDA(I),J)
             ! exclude non-bond exclusions
             pario_qexcl=QHBEXCL(IATOM,JJ,IBLO14,INB14)
             IF(pario_qexcl) cycle no_bridge
             ! Not excluded from topology
             QCLOSE=QDISANG(IATOM,JJ,CUT,CUTA,DFLAG(IATOM),HEAVY,QVDWR)
#if _OPENMP
             pario_qclose(j)=qclose ! store it here for later scalar processing
#else
             ! for OpenMP process this if block in a scalar loop below:
            IF(QCLOSE)THEN
               HBNO(I)=HBNO(I)+1
               ! Is this someone we already know?
               L=0
               DO K=1,MAXDA
                  IF(JJ  ==  HBLST(K,I))THEN
                     HBAGE(K,I)=HBAGE(K,I)+1
                     cycle no_bridge
                  ENDIF
               ENDDO
               !
               ! Find a slot to add this one to
               DO K=1,MAXDA
                  IF(HBLST(K,I)  ==  0)THEN
                     HBLST(K,I)=JJ
                     HBAGE(K,I)=1
                     cycle no_bridge
                  ENDIF
               ENDDO
               !
               CALL WRNDIE(0,'<HBANAL>','HBLST DIMENSION EXCEEDED')
               call hban1_return
               RETURN
            ENDIF
110          CONTINUE
#endif
         ENDDO no_bridge

#if _OPENMP
          !$OMP END DO 
          !$OMP END PARALLEL

          ! Perform the scalar part of the above loop
          no_bridge_scalar: DO J=1,pario_NPAIR
             JJ=JDALST(1-IDA(I),J)
             IF(pario_QCLOSE(j))THEN
                HBNO(I)=HBNO(I)+1
                ! Is this someone we already know?
                L=0
                DO K=1,MAXDA
                   IF(JJ  ==  HBLST(K,I))THEN
                      HBAGE(K,I)=HBAGE(K,I)+1
                      cycle no_bridge_scalar
                   ENDIF
                ENDDO
                !
                ! Find a slot to add this one to
                DO K=1,MAXDA
                   IF(HBLST(K,I)  ==  0)THEN
                      HBLST(K,I)=JJ
                      HBAGE(K,I)=1
                      cycle no_bridge_scalar
                   ENDIF
                ENDDO
                !
                CALL WRNDIE(0,'<HBANAL>','HBLST DIMENSION EXCEEDED')
                call hban1_return
                RETURN
             ENDIF
          enddo no_bridge_scalar

          deallocate(pario_qclose)
#endif
       ELSE book_keep1
          !
          ! For each I-atom, now look at possible hbonds to bridge molecules
          ! and when one found, see if this ALSO hbonds to a J-atom
          II=ISLCT(I)
          DO J=1,NBRDG(1-IDA(I))
             JJ=BDALST(1-IDA(I),J)
             ! And, exclude non-bond exclusions
             IF(QHBEXCL(II,JJ,IBLO14,INB14)) GOTO 230
             ! Not excluded from topology
             QCLOSE=QDISANG(II,JJ,CUT,CUTA,DFLAG(II),HEAVY,QVDWR)
             IF(QCLOSE)THEN
                ! Look at all acceptors in this bridge molecule
                KK=IBASE(GETRES(JJ,IBASE,NRES) ) +1
                DO LL=1,NBA
                   K=KK+OFFS(ACCEPTOR,LL)
                   ! ...and see if it is within distance to any donor in 2nd set
                   DO L=1,NPAIR(DONOR)
                      M=JDALST(DONOR,L)
                      IF(QHBEXCL(K,M,IBLO14,INB14).OR. M == II) &
                           GOTO 210
                      QCLOSE=QDISANG(M,K,CUT,CUTA,DFLAG(M),HEAVY,QVDWR)
                      IF(QCLOSE)THEN
                         HBNO(I)=HBNO(I)+1
                         ! An old friend?
                         DO MM=1,MAXDA
                            IF(M == HBLST(MM,I))THEN
                               HBAGE(MM,I)=HBAGE(MM,I)+1
                               GOTO 210
                            ENDIF
                         ENDDO
                         ! So we find an empty slot
                         DO MM=1,MAXDA
                            IF(HBLST(MM,I) == 0)THEN
                               HBLST(MM,I)=M
                               HBAGE(MM,I)=1
                               HBRDG(MM,I)= &
                                    IBASE(GETRES(K,IBASE,NRES))+1
                               GOTO 210
                            ENDIF
                         ENDDO
                         CALL WRNDIE(1,'<HBANAL>', &
                              'HBLST DIMENSION EXCEEDED')
                         call hban1_return
                         RETURN
                      ENDIF
210                   CONTINUE
                   ENDDO
                ENDDO
                ! let's also look at all donors in this bridge molecule
                DO LL=1,NBD
                   K=KK+OFFS(DONOR,LL)
                   ! ...and see what acceptors in 2nd set are close
                   DO L=1,NPAIR(ACCEPTOR)
                      M=JDALST(ACCEPTOR,L)
                      IF(QHBEXCL(K,M,IBLO14,INB14).OR.M == II) &
                           GOTO 220
                      QCLOSE=QDISANG(K,M,CUT,CUTA,DFLAG(K),HEAVY,QVDWR)
                      IF(QCLOSE)THEN
                         HBNO(I)=HBNO(I)+1
                         ! An old friend?
                         DO MM=1,MAXDA
                            IF(M == HBLST(MM,I))THEN
                               HBAGE(MM,I)=HBAGE(MM,I)+1
                               GOTO 220
                            ENDIF
                         ENDDO
                         ! So we find an empty slot
                         DO MM=1,MAXDA
                            IF(HBLST(MM,I) == 0)THEN
                               HBLST(MM,I)=M
                               HBAGE(MM,I)=1
                               HBRDG(MM,I)= &
                                    IBASE(GETRES(K,IBASE,NRES))+1
                               GOTO 220
                            ENDIF
                         ENDDO
                         CALL WRNDIE(1,'<HBANAL>', &
                              'HBLST DIMENSION EXCEEDED')
                         call hban1_return
                         RETURN
                      ENDIF
220                   CONTINUE
                   ENDDO
                ENDDO
             ENDIF
230          CONTINUE
          ENDDO
       ENDIF book_keep1
    ENDDO loop1
    ! Get a new coordinate set, or wrap up
#if KEY_PARALLEL==1
    ! this might help to organize output better:
    if (pario_q) then
       call mpi_barrier(mpi_comm_world,ll)
    endif
#endif
    IF(.NOT. DONE) GOTO 100
#if KEY_PARALLEL==1
    if (pario_q) then
!       call mpi_allreduce(mpi_in_place,hbno,isel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       call mpi_allreduce(mpi_in_place,hblst,maxda*isel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       call mpi_allreduce(mpi_in_place,hbage,maxda*isel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       if(qbridge) then
          call mpi_allreduce(mpi_in_place,hbrdg,maxda*isel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       endif
    endif
#endif
    ! for parallel this loop needs to be the same as loop1
#if KEY_PARALLEL==1
    if(.not.pario_q) then
       pario_st=1
       pario_sk=1
    else
       pario_st=mynodp
       pario_sk=numnod
    endif
    loop2: DO I=pario_st,ISEL,pario_sk
#else
    loop2: DO I=1,ISEL
#endif
#if KEY_PARALLEL==1
       pario_count=0
#endif
       DO J=1,MAXDA
          IF(HBLST(J,I)  >  0)THEN
             PS=HBAGE(J,I)*STPTIME
             ! Lifetime histogram
             IT1=MIN(MXTH,INT(PS/DTH))
             THIST(IT1)=THIST(IT1)+1
             IF(PS > TCUT)THEN
                HBLIFE(I)=HBLIFE(I)+HBAGE(J,I)
                HBCNT(I)=HBCNT(I)+1
                IF(QVERB .AND. NSTEP > 1)THEN
                   JJ=HBLST(J,I)
                   IATOM=ISLCT(I)
                   CALL ATOMID(IATOM,SID,RID,REN,IUPAC)
                   CALL ATOMID(JJ,SID2,RID2,REN2,IUPAC2)
                   IF(QBRIDGE)THEN
                      KK=HBRDG(J,I)
                      CALL ATOMID(KK,SID3,RID3,REN3,IUPAC3)
#if KEY_PARALLEL==1
                      if(mynod==0) then
#endif
                         WRITE(IUNIT,901) SID(1:idleng),RID(1:idleng), &
                              REN(1:idleng),IUPAC(1:idleng), &
                              SID2(1:idleng),RID2(1:idleng), &
                              REN2(1:idleng),IUPAC2(1:idleng), &
                              SID3(1:idleng),RID3(1:idleng), &
                              PS,NSTEP*STPTIME
#if KEY_PARALLEL==1
                      else if (pario_q) then
                         pario_count = pario_count +1
                         WRITE(pario_lines(pario_count),901) &
                              SID(1:idleng),RID(1:idleng), &
                              REN(1:idleng),IUPAC(1:idleng), &
                              SID2(1:idleng),RID2(1:idleng), &
                              REN2(1:idleng),IUPAC2(1:idleng), &
                              SID3(1:idleng),RID3(1:idleng), &
                              PS,NSTEP*STPTIME
                      endif
#endif
                   ELSE
#if KEY_PARALLEL==1
                      if(mynod==0) then
#endif
                         WRITE(IUNIT,901) SID(1:idleng),RID(1:idleng), &
                              REN(1:idleng),IUPAC(1:idleng), &
                              SID2(1:idleng),RID2(1:idleng), &
                              REN2(1:idleng),IUPAC2(1:idleng),' ',' ', &
                              PS,NSTEP*STPTIME
#if KEY_PARALLEL==1
                      else if (pario_q) then
                         pario_count = pario_count +1
                         WRITE(pario_lines(pario_count),901) &
                              SID(1:idleng),RID(1:idleng), &
                              REN(1:idleng),IUPAC(1:idleng), &
                              SID2(1:idleng),RID2(1:idleng), &
                              REN2(1:idleng),IUPAC2(1:idleng),' ',' ', &
                              PS,NSTEP*STPTIME
                      endif
#endif
                   ENDIF
                   J2=JSEL2(JJ)
                   LHBLIST(1,J2,I)=LHBLIST(1,J2,I) + HBAGE(J,I)
                   LHBLIST(2,J2,I)=LHBLIST(2,J2,I)+1
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       !
#if KEY_PARALLEL==1
       ! for parallel communicate the written lines for each atom
       ! with additional bookkeeping this block can be moved after the loop2
       ! to prevent blocking first communicate pario_count-s
       if(pario_q) then
          pario_tag=99
          if (mynod==0) then
             do pario_i=1,numnod-1
                call mpi_recv(pario_count_all(pario_i),1,mpi_integer, &
                     pario_i,pario_tag, mpi_comm_world, pario_stat, pario_err)
             end do
          else
             call mpi_send(pario_count,1,mpi_integer,0, &
                  pario_tag,mpi_comm_world,pario_err)
          endif
          ! now process the output lines... no order here /maybe it comes later??/ :-(
          if (mynod==0) then
             do pario_i=1,numnod-1
                if (pario_count_all(pario_i) > 0) then
                   call mpi_recv(pario_lines,pario_len*pario_count_all(pario_i), &
                        mpi_character, pario_i, pario_tag, mpi_comm_world, &
                        pario_stat, pario_err)
                   ! this could be TRIMmed but then we probably need a loop here?
                   write(outu,'(a)')pario_lines(1:pario_count_all(pario_i))(1:pario_len)
                endif
             end do
          else
             if(pario_count >0) call mpi_send(pario_lines,pario_len*pario_count, &
                  mpi_character,0,pario_tag,mpi_comm_world,pario_err)
          endif
       endif
#endif
       !
    ENDDO loop2
#if KEY_PARALLEL==1
    ! barrier not needed due to global sum:
    ! call mpi_barrier(mpi_comm_world,ll) ! to make output more uniform
    if(pario_q)then
       call mpi_allreduce(mpi_in_place,hblife,isel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       call mpi_allreduce(mpi_in_place,hbcnt,isel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       call mpi_allreduce(mpi_in_place,thist,mxth,mpi_integer,mpi_sum,mpi_comm_world,ll)
       if(qverb)call mpi_allreduce(mpi_in_place,lhblist,2*isel*jsel,mpi_integer,mpi_sum,mpi_comm_world,ll)
       deallocate(pario_lines,pario_count_all)
    endif
#endif
    !
    ! Output results
    ! Convert minimum occupancy fraction (PCUT) to minimum number of steps
    MINSTEP=PCUT*NSTEP
    !
    ! Let the summary information goto standard output OUTU

    IF(QVERB.AND.NSTEP == 1)THEN
       IF(QBRIDGE)THEN
          if(prnlev>=2)WRITE(OUTU,890) BRIDGE,LEADTXT,CUT,CUTA
       ELSE
          if(prnlev>=2)WRITE(OUTU,891) LEADTXT,CUT,CUTA
       ENDIF
890    FORMAT(/' Analysis of ',A,1X,A, &
            ' bridges, using cutoff distance= ',F7.2,' and angle=',F7.2, &
            /5X,'I-atom',17X,'Bridge',7X,'J-atom',/, &
            ' -----------------------------------------------------------', &
            '---------')

891    FORMAT(/' Analysis of ',A,'s using cutoff distance= ', &
            F7.2,' and angle=',F7.2, &
            /5X,'I-atom',17X,'J-atom',12X,'r(H-A) [A]  D-H..A [deg]',/, &
            ' -----------------------------------------------------------', &
            '---------')
    ELSE IF(QVERB .AND. NSTEP > 1)THEN
       IF(QBRIDGE)THEN
          if(prnlev>=2)WRITE(OUTU,895) BRIDGE,LEADTXT,CUT,CUTA,NSTEP*STPTIME, &
               STPTIME,PCUT,TCUT
       ELSE
          if(prnlev>=2)WRITE(OUTU,896) LEADTXT,CUT,CUTA,NSTEP*STPTIME, &
               STPTIME,PCUT,TCUT
       ENDIF
895    FORMAT(/' Analysis of ',A,1X,A, &
            ' bridges, using cutoff distance= ',F7.2,' and angle=',F7.2, &
            /,' Total time=',F9.1,'ps.  Resolution=',F7.2,'ps', &
            /,' Occupancy cutoff=',F5.2,'  Lifetime cutoff=',F6.1,'ps',/, &
            /5X,'I-atom',17X,'J-atom',10X,'<occupancy>     <time>  # events', &
            /' -----------------------------------------------------------', &
            '----------------')

896    FORMAT(/' Analysis of ',A,'s using cutoff distance= ', &
            F7.2,' and angle=',F7.2, &
            /,' Total time=',F9.1,'ps.  Resolution=',F7.2,'ps', &
            /,' Occupancy cutoff=',F5.2,'  Lifetime cutoff=',F6.1,'ps',/, &
            /5X,'I-atom',17X,'J-atom',10X,'<occupancy>     <time>  # events', &
            /' -----------------------------------------------------------', &
            '----------------')

    ELSE
       IF(QBRIDGE) THEN
          if(prnlev>=2)WRITE(OUTU,900) BRIDGE,LEADTXT,' bridges',NSTEP*STPTIME, &
               STPTIME
       ELSE
          if(prnlev>=2)WRITE(OUTU,900) ' ',LEADTXT,'s',NSTEP*STPTIME,STPTIME
       ENDIF
       IF(CUTA  <=  THR6TY)THEN
          if(prnlev>=2)WRITE(OUTU,905) CUT,CUTA,TCUT,PCUT
       ELSE
          if(prnlev>=2)WRITE(OUTU,906) CUT,TCUT,PCUT
       ENDIF
    ENDIF
900 FORMAT(//1X,A,1X,A,A,'. Total time=',F9.1,'ps. Resolution=', &
         F7.3,'ps.')
905 FORMAT(' Cutoff distance=',F7.2,' angle=',F7.2,' time=',F7.1, &
         ' occupancy=',F5.2, &
         /10X,'ATOM',12X,'<occupancy>',6X,'<lifetime> (ps)')
906 FORMAT(' Cutoff distance=',F7.2,' time=',F7.1, &
         ' occupancy=',F5.2, &
         /10X,'ATOM',12X,'<occupancy>',6X,'<lifetime> (ps)')
    AVSUM=ZERO
    TSUM=ZERO
    MM=0
    DO I=1,ISEL
       IATOM=ISLCT(I)
       CALL ATOMID(IATOM,SID,RID,REN,IUPAC)
       AVNO=HBNO(I)
       AVLIFE=HBLIFE(I)*STPTIME
       IF(NSTEP  > 1 .AND. QVERB)THEN
          AVINO=ZERO
          AVILIF=ZERO
          IEV=0
          M=0
          DO J=1,JSEL
             K=LHBLIST(2,J,I)
             L=LHBLIST(1,J,I)
             IF(K > 0 .AND. L >= MINSTEP)THEN
                AVLIFE1=L*STPTIME/K
                IF(AVLIFE1 > TCUT)THEN
                   AVNO=FLOAT(L)/FLOAT(NSTEP)
                   AVINO=AVINO+AVNO
                   AVILIF=AVILIF+AVLIFE1
                   IEV=IEV+K
                   M=M+1
                   JJ=JSLCT(J)
                   CALL ATOMID(JJ,SID2,RID2,REN2,IUPAC2)
                   if(prnlev>=2)WRITE(OUTU,908) SID(1:idleng),REN(1:idleng), &
                        RID(1:idleng),IUPAC(1:idleng), &
                        SID2(1:idleng),REN2(1:idleng), &
                        RID2(1:idleng),IUPAC2(1:idleng), &
                        AVNO,AVLIFE1,K
                ENDIF
             ENDIF
          ENDDO
          IF(M > 0)THEN
             AVINO=AVINO   ! strange statement, is something amiss here? /LNI
             AVSUM=AVSUM+AVINO
             AVILIF=AVILIF/M
             TSUM=TSUM+AVILIF
             MM=MM+1
             IF(M > 1)THEN
                if(prnlev>=2)WRITE(OUTU,909) SID(1:idleng),REN(1:idleng), &
                     RID(1:idleng),IUPAC(1:idleng),AVINO,AVILIF,IEV
             ENDIF
909          FORMAT(15X,4(1X,A),5X,'Summary:',F7.3,4X,F7.1,I8)
          ENDIF
       ELSE IF(NSTEP == 1 .AND. QVERB)THEN
          DO K=1,MAXDA
             J=HBLST(K,I)
             IF(J  /=  0)THEN
                HBDEFI(IATOM)=1
                HBDEFJ(J)=1 
                CALL ATOMID(J,SID2,RID2,REN2,IUPAC2)
                IF(QBRIDGE)THEN
                   L=HBRDG(K,I)
                   HBDEFB(L)=1
                   CALL ATOMID(L,SID3,RID3,REN3,IUPAC3)
                   if(prnlev>=2)WRITE(OUTU,907) SID(1:idleng),REN(1:idleng), &
                        RID(1:idleng),IUPAC(1:idleng), &
                        SID3(1:idleng),RID3(1:idleng), &
                        SID2(1:idleng),REN2(1:idleng), &
                        RID2(1:idleng),IUPAC2(1:idleng)
                ELSE
                   IF(DFLAG(IATOM))THEN
                      ID=IATOM
                      IA=J
                   ELSE
                      ID=J
                      IA=IATOM
                   ENDIF
                   L=HEAVY(ID)
                   CALL ATDANG(IA,ID,L,X,Y,Z,HBDIST,HBANG)
                   if(prnlev>=2)WRITE(OUTU,908) SID(1:idleng),REN(1:idleng), &
                        RID(1:idleng),IUPAC(1:idleng), &
                        SID2(1:idleng),REN2(1:idleng), &
                        RID2(1:idleng),IUPAC2(1:idleng), &
                        HBDIST,HBANG
                ENDIF
907             FORMAT(5X,4(1X,A),' - ',2(1X,A),' - ',4(1X,A))
908             FORMAT(5X,4(1X,A),' - ',4(1X,A),F7.3,4X,F7.1,I8)
             ENDIF
          ENDDO
       ELSE
          IF(NSTEP  ==  1)THEN
             if(prnlev>=2)WRITE(OUTU,910) SID(1:idleng),REN(1:idleng), &
                  RID(1:idleng),IUPAC(1:idleng),AVNO
             AVSUM=AVSUM+AVNO
          ELSEIF(AVLIFE > ZERO)THEN
             AVLIFE=AVLIFE/HBCNT(I)
             IF(AVLIFE  >=  TCUT)THEN
                AVNO=AVNO/NSTEP
                if(prnlev>=2)WRITE(OUTU,910) SID(1:idleng),REN(1:idleng), &
                     RID(1:idleng),IUPAC(1:idleng),AVNO,AVLIFE
                AVSUM=AVSUM+AVNO
                TSUM=TSUM+AVLIFE
                MM=MM+1
             ENDIF
          ENDIF
       ENDIF
910    FORMAT(4(1X,A),5X,2F12.4)
    ENDDO
    ! Print out summation, and set misc param values
    IF(NSTEP == 1 .AND. QVERB)THEN
      LEADTXT='HBDEFI'
      I=6
      CALL FILSKY(LEADTXT,I,LUSED,.TRUE.,HBDEFI)
      LEADTXT='HBDEFJ'
      I=6
      CALL FILSKY(LEADTXT,I,LUSED,.TRUE.,HBDEFJ)
      IF(QBRIDGE)THEN
        LEADTXT='HBDEFB'
        I=6
        CALL FILSKY(LEADTXT,I,LUSED,.TRUE.,HBDEFB)
      ENDIF
    ENDIF  
    call set_param('NHBOND',AVSUM)
    IF(ISEL > 0)  AVSUM=AVSUM/ISEL
    IF(MM > 0)    TSUM=TSUM/MM
    call set_param('AVNOHB',AVSUM)
    call set_param('AVHBLF',TSUM)
    IF(NSTEP > 1)THEN
       if(prnlev>=2)WRITE(OUTU,911) AVSUM,TSUM,AVSUM*ISEL
911    FORMAT(' -----------------------------------------------------', &
            /' OVERALL', &
            /' Average number (over ISEL!)=',F7.3, &
            /' <lifetime>=',F10.2,'ps', &
            /' Total number/frame=',F10.1/)
    ENDIF
    if(prnlev>=2) then
       ! Histograms
       XSTEP=NSTEP 
       IF(IUNRH == OUTU) WRITE(OUTU,600)
600    FORMAT(//' Histogram for all hydrogen-acceptor distances (A)')
       IF(IUNRH > 0)THEN
          WRITE(IUNRH,'(A)') '      R      RCOUNT    RC/NSTEP'
          WRITE(IUNRH,'(F8.3,I12,F14.5)') &
               ((I+0.5)*DRH,IRHIST(I),IRHIST(I)/XSTEP,I=0,MXRH)
       ENDIF
       IF(IUNTH == OUTU) WRITE(OUTU,601)
601    FORMAT(//' Histogram for all hydrogen-bond lifetimes (ps)')
       IF(IUNTH > 0)THEN
          WRITE(IUNTH,'(A)') '      T      TCOUNT     TC/NSTEP'
          WRITE(IUNTH,'(F8.2,I12,F14.5)')  &
               ((I+0.5)*DTH,THIST(I),THIST(I)/XSTEP,I=0,MXTH)
       ENDIF
    endif
    !
    call hban1_return
    RETURN

  contains

    subroutine hban1_return

      call chmdealloc('hbanal.src','HBAN1','ispace',4*isel+2*natom,intg=ispace)
      call chmdealloc('hbanal.src','HBANAL','LHBLIST', &
           2,lhblist_siz1,lhblist_siz2,intg=LHBLIST)
      call chmdealloc('hbanal.src','HBANAL','JSEL2',jsel2_siz,intg=JSEL2)

      call chmdealloc('hbanal.src','HBAN1','temp',natom,cr4=temp)
      call chmdealloc('hbanal.src','HBAN1','HBLST',MAXDA,ISEL,intg=hblst)
      call chmdealloc('hbanal.src','HBAN1','HBAGE',MAXDA,ISEL,intg=hbage)
      call chmdealloc('hbanal.src','HBAN1','JDALST',2,mxjda,intg=jdalst)

      call chmdealloc('hbanal.src','HBAN1','HBRDG', &
           hbrdgsz1,hbrdgsz2,intg=hbrdg)
      call chmdealloc('hbanal.src','HBAN1','BDALST', &
           bdalstsz1,bdalstsz2,intg=bdalst)

      IF(QVERB)THEN
        call chmdealloc('hbanal.src','HBAN1','HBDEFI',natom,intg=hbdefi)
        call chmdealloc('hbanal.src','HBAN1','HBDEFJ',natom,intg=hbdefj)
        IF(QBRIDGE) call chmdealloc('hbanal.src','HBAN1','HBDEFB',natom,intg=hbdefb)
      ENDIF
      return
    end subroutine hban1_return

  END SUBROUTINE HBAN1
  !
  !  SUBROUTINE MAKIND(INDEX,NATOM,NIND)
  !-----------------------------------------------------------------------
  ! Transforms the array INDEX into a new array which contains
  ! the indices of all elements (NIND) with INDEX(i) == 1.
  !
  ! Author: Axel Brunger, 21-AUG-84
  !
  ! Imported from Charmm_19 by Arne Elofsson
  !

  ! input/output
  !    INTEGER   NATOM, NIND
  !    INTEGER INDEX(*)
  ! local
  !    INTEGER I
  ! begin
  !    NIND=0
  !    DO I=1,NATOM
  !       IF(INDEX(I) == 1)THEN
  !          NIND=NIND+1
  !          INDEX(NIND)=I
  !       ENDIF
  !    ENDDO
  !    RETURN
  !  END SUBROUTINE MAKIND

  LOGICAL FUNCTION QHBEXCL(I,J,IBLO14,INB14)
    !-----------------------------------------------------------------------
    ! Check if I and J are bonded together and therefore should not
    ! be allowed to form H-bond
    !
    INTEGER I,J,IBLO14(*),INB14(*)
    INTEGER IS,IQ,K
    !
    QHBEXCL=.TRUE.
    IF(I == 1) THEN
       IS=1
    ELSE
       IS=IBLO14(I-1)+1
    ENDIF
    IQ=IBLO14(I)
    DO K=IS,IQ
       IF(IABS(INB14(K)) == J) RETURN
    ENDDO
    ! Check  list both ways....
    IF(J == 1) THEN
       IS=1
    ELSE
       IS=IBLO14(J-1)+1
    ENDIF
    IQ=IBLO14(J)
    DO K=IS,IQ
       IF(IABS(INB14(K)) == I) RETURN
    ENDDO
    !  we could have the same atom here twice for some reason...
    IF(I  ==  J) RETURN
    !
    ! passed all checks, so no need to exclude this pair
    QHBEXCL=.FALSE.
    RETURN
  END FUNCTION QHBEXCL

  LOGICAL FUNCTION QDISANG(I,J,CUT,CUTA,QIDON,HEAVY,QVDWR)
    !-----------------------------------------------------------------------
    use exfunc
    use number

    INTEGER I,J,HEAVY(*)
    LOGICAL QIDON,QVDWR
    real(chm_real) CUT,CUTA
    !

    ! Just find out which of I and J is the donor, and which one is the acceptor
    !
    IF(QIDON)THEN
       QDISANG=QDISA2(I,J,HEAVY(I),CUT,CUTA,QVDWR)
    ELSE
       QDISANG=QDISA2(J,I,HEAVY(J),CUT,CUTA,QVDWR)
    ENDIF
    RETURN
  END FUNCTION QDISANG

  LOGICAL FUNCTION QDISA2(ID,IA,IH,CUT,CUTA,QVDWR)
    !-----------------------------------------------------------------------
    !     True if atoms indexed by ID (donor hydrogen) and IA (acceptor)
    !     are within CUT distance and CUTA angle.
    !     IH is the  atom (donor) to which Hydrogen is
    !     covalently bonded, and is only used if angle has to be evaluated
    !     CUT is the cutoff distance and CUTA is
    !     the cutoff angle (or some large value>360 if the angle criterion
    !     is not to be used)
    !
    use exfunc
    use number
    use coord
    use psf,only: iac
    use param,only: vdwr,itc
    INTEGER ID,IA,IH
    logical QVDWR
    real(chm_real) CUT, CUTA, DIST, ANGL, CUTOFF
    INTEGER I
    ! 
    CUTOFF=CUT
    IF(QVDWR) THEN
      ! use sum of vdW-radii of acceptor and donor(heavy) atom, modified by CUT
      CUTOFF=VDWR(ITC(IAC(IA))) + VDWR(ITC(IAC(IH))) + CUT
    ENDIF
    IF(CUTA < THR6TY)THEN
       CALL ATDANG(IA,ID,IH,X,Y,Z,DIST,ANGL)
       QDISA2= (DIST <=  CUTOFF) .AND. (ANGL  >  CUTA)
    ELSE
       CALL ATDANG(IA,ID,ID,X,Y,Z,DIST,ANGL)
       QDISA2= (DIST <=  CUTOFF)
    ENDIF
    ! Accumulate histogram, but only if this has been setup
    IF(.NOT. QHBHIST) RETURN
    I=MIN(MXRH, INT(DIST/DRH))
    IRHIST(I)=IRHIST(I)+1 
    RETURN
  END FUNCTION QDISA2
  !
  SUBROUTINE ATDANG(I,J,K,X,Y,Z,DIST,ANGL)
    !-----------------------------------------------------------------------
    use exfunc
    use number
    use consta
    use stream

    !
    ! Assuming that atoms exist!
    ! compute distance atom(I)-atom(J) and angle (degrees) formed by
    ! atoms I-J-K (if J  /=  K)
    INTEGER I,J,K
    real(chm_real) X(*),Y(*),Z(*),DIST,ANGL
    !
    real(chm_real) DX,DY,DZ,GX,GY,GZ,CST,GR
    !
    ANGL=ZERO
    DIST=ZERO
    IF(I == J) RETURN
    DX=X(I)-X(J)
    DY=Y(I)-Y(J)
    DZ=Z(I)-Z(J)
    DIST=SQRT(DX*DX+DY*DY+DZ*DZ)
#if KEY_PBOUND==1
    CALL MI_DIST(DX,DY,DZ,DIST)                  
#endif
    IF(K == J) RETURN
    GX=X(K)-X(J)
    GY=Y(K)-Y(J)
    GZ=Z(K)-Z(J)
    GR=SQRT(GX*GX+GY*GY+GZ*GZ)
#if KEY_PBOUND==1
    CALL MI_DIST(GX,GY,GZ,GR)                    
#endif
    IF(GR+DIST  <=  TENM8)THEN
       IF(WRNLEV  >=  2 .AND. PRNLEV  >=  2) THEN
          WRITE(OUTU,*) '%%%-ATDANG: No valid angle, returning 0.0'
          WRITE(OUTU,*)
       ENDIF
       RETURN
    ENDIF
    CST=(DX*GX+DY*GY+DZ*GZ)/GR/DIST
    !
    IF(ABS(CST) >= COSMAX) THEN
       ANGL=NINETY-SIGN(NINETY,CST)
    ELSE
       ANGL=ACOS(CST)*RADDEG
    ENDIF
    RETURN
  END SUBROUTINE ATDANG
  !
#if KEY_PBOUND==1
  SUBROUTINE MI_DIST(TX,TY,TZ,DIST)
    !--------------------------------------
    ! 
    ! Compute Minimum Image distance DIST and correct the DX,DY,DZ accordingly
    ! L. Nilsson, KI, March 2005, with input from G. Lamoureux, Cornell Med School
    !
    use number
    use pbound
    use exfunc

    real(chm_real) TX,TY,TZ,DIST
    !
    real(chm_real) CORR
    !
    IF(.NOT. QBOUN) RETURN
    ! Apply corrections as needed
    If(qCUBoun.or.qTOBoun) then
       TX      = BOXINV * TX
       TY      = BOYINV * TY
       TZ      = BOZINV * TZ
       TX = TX - ANINT(TX)
       TY = TY - ANINT(TY)
       TZ = TZ - ANINT(TZ)
       !            IF(TX >  HALF) TX = TX - ONE
       !            IF(TX <  -HALF) TX = TX + ONE
       !            IF(TY >  HALF) TY = TY - ONE
       !            IF(TY <  -HALF) TY = TY + ONE
       !            IF(TZ >  HALF) TZ = TZ - ONE
       !            IF(TZ <  -HALF) TZ = TZ + ONE
       IF(qTOBoun) Then
          CORR = HALF * AINT ( R75 * ( ABS ( TX ) + &
               ABS ( TY ) + &
               ABS ( TZ ) ) )
          TX      = TX    - SIGN ( CORR,  TX  )
          TY      = TY    - SIGN ( CORR,  TY  )
          TZ      = TZ    - SIGN ( CORR,  TZ  )
       Endif
       TX      = XSIZE * TX
       TY      = YSIZE * TY
       TZ      = ZSIZE * TZ
    Else
       Call PBMove(TX, TY, TZ)
    Endif
    DIST=SQRT(MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)) 
    RETURN
  END SUBROUTINE MI_DIST
#endif 

end module hbanal_mod

