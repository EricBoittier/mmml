module nbndcc_util
!
contains

  SUBROUTINE BOX27
  use chm_kinds
  use dimens_fcm
  use actclus_mod,only:bbx,bby,bbz
    implicit none
    !  generates a 27-row matrix of coordinates
    !  corresponding to positions of each cube
    !  in a 3x3 box of 27 cubes, where
    !  (X,Y,Z) = (0,0,0) corresponds to the
    !  position of the center cube.
    !   -RJ Petrella 1.6.99
    !
    INTEGER I,J,K,C
    C = 0
    DO I = -1,1
       DO J = -1,1
          DO K = -1,1
             C = C + 1
             BBX(C) = I
             BBY(C) = J
             BBZ(C) = K
          ENDDO
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE BOX27

  SUBROUTINE NBACTV(COMLYN,COMLEN)
    !----------------------------------------
    ! This routine generates the arrays of active
    ! atoms (ACTV),  groups (ACTVG), and clusters
    ! (ACLHI/JACLS).
    ! Note that the requirement for either a group
    ! or a cluster to be active is for it to contain
    ! at least one active atom.
    ! However, groups and clusters are treated
    ! differently in that groups are treated as single
    ! particles: a group is simply either active
    ! or inactive.
    ! In contrast, an active cluster may contain both
    ! active and inactive atoms.
    ! An inactive cluster contains no active atoms.
    !   - RJ Petrella 1.6.99
    !
  use chm_kinds
  use dimens_fcm
  use coord
  use psf
  use actclus_mod
  use select
  use stream
  use machutil,only:die
  use memory
    implicit none
    ! . Passed variables.
    CHARACTER COMLYN*(*)
    INTEGER   COMLEN
    !
    !. Local var
    INTEGER I,J,TT,CNT
    INTEGER AAA,IS,IQ,NAT
    INTEGER RCOUNT,BB,ACC,II

    integer, allocatable, dimension(:) :: flact
    integer :: actcat,alloc_err
    integer,save :: sizea

    LOGICAL INCLU
    ! End of variable declarations
    ! check to see whether clusters are made
    IF(.NOT.QCLSMD) CALL MKCLUSTDEF
    !

    allocate(flact(natom),stat=alloc_err)
    if(alloc_err /= 0)then
       if(prnlev >= 2) write(outu,'(a)') &
            "cannot allocate array in nbutil.src"
       CALL WRNDIE(0,'<NBACTV nbutil.src>','allocate returns error')
    endif

    !     Added initialization of FLACT --RJP 4.15.99
    DO I = 1,NATOM
       FLACT(I) = 0
       ! initialize active atom flag array --RJP 10.03
       ACAFLG(I) = 0
    ENDDO
    ! -----------------------------------------------
    IF(PRNLEV >= 2) WRITE (OUTU, '(A)') &
         ' CREATING ACTIVE ARRAYS '
    CALL SELCTA(COMLYN,COMLEN,FLACT,X,Y,Z,WMAIN,.TRUE.)

! deallocate if previously allocated
    if(allocated(ACTV)) then
     call chmdealloc('nbndcc_util','ACTVDEF','ACTV',sizea,intg=ACTV)
     call chmdealloc('nbndcc_util','ACTVDEF','ACAFLG',NATOM,intg=ACAFLG)
     call chmdealloc('nbndcc_util','ACTVDEF','JACLS',sizea,intg=JACLS)
     call chmdealloc('nbndcc_util','ACTVDEF','ACLPTN',CLNUM,intg=ACLPTN)
     call chmdealloc('nbndcc_util','ACTVDEF','ACLHI',CLNUM,intg=ACLHI)
     call chmdealloc('nbndcc_util','ACTVDEF','ACTVG',NGRP,intg=ACTVG)
     call chmdealloc('nbndcc_util','ACTVDEF','ACGFLG',NGRP,intg=ACGFLG)
    endif

    sizea = 0
    DO II = 1,NATOM
     if(FLACT(II) > 0) THEN
       sizea = sizea + 1
     endif
    ENDDO

    call chmalloc('nbndcc_util','ACTVDEF','ACTV',sizea,intg=ACTV)
    call chmalloc('nbndcc_util','ACTVDEF','ACAFLG',NATOM,intg=ACAFLG)
    call chmalloc('nbndcc_util','ACTVDEF','JACLS',sizea,intg=JACLS)
    call chmalloc('nbndcc_util','ACTVDEF','ACLPTN',CLNUM,intg=ACLPTN)
    call chmalloc('nbndcc_util','ACTVDEF','ACLHI',CLNUM,intg=ACLHI)
    call chmalloc('nbndcc_util','ACTVDEF','ACTVG',NGRP,intg=ACTVG)
    call chmalloc('nbndcc_util','ACTVDEF','ACGFLG',NGRP,intg=ACGFLG)

    !    create active atom array
    NACTVE = 0
    DO I=1,NATOM
       IF (FLACT(I) > 0) THEN
          ACAFLG(I) = 1
          NACTVE = NACTVE + 1
          ACTV(NACTVE) = I
       ENDIF
    ENDDO
    IF (NACTVE == 0) CALL WRNDIE(1,'<NBACTV>', &
         'NO ACTIVE ATOMS SELECTED')
    !  create active cluster array
    ACTCAT = 0
    NACTC = 0
    DO I=1,CLNUM
       IQ = CLHIGH(I)
       IS = IQ - CLPRTN(I) + 1
       NAT=CLPRTN(I)
       IF(NAT <= 0) CALL DIE
       INCLU =.FALSE.
       CNT = 0
       DO TT = IS,IQ
          J = JCLUS(TT)
          IF (FLACT(J) > 0) THEN
             IF(.NOT.INCLU) THEN
                NACTC = NACTC + 1
                INCLU = .TRUE.
             ENDIF
             CNT = CNT + 1
             ACTCAT = ACTCAT + 1
             JACLS(ACTCAT) = J
          ENDIF
       ENDDO
       IF (INCLU) THEN
          ACLPTN(NACTC) = CNT
          ACLHI(NACTC) = ACTCAT
       ENDIF
    ENDDO
    !   create active group array
    ! Added ACGFLG initialization just below--RJP
    ! First initialize active group flag array
    DO I = 1,NGRP
       ACGFLG(I) = 0
    ENDDO
    ! fill ACGFLG array and ACTVG array
    NACTG = 0
    DO I=1,NGRP
       IS=IGPBS(I)+1
       IQ=IGPBS(I+1)
       NAT=IQ-IS+1
       IF (NAT <= 0) CALL DIE
       TT = IS
       INCLU = .FALSE.
       CNT = 0
       DO J = IS,IQ
          IF (FLACT(J) > 0) THEN
             IF(.NOT.INCLU) THEN
                NACTG = NACTG + 1
                !    Added 1 line below -RJP
                ACGFLG(I) = 1
                ACTVG(NACTG) = I
                INCLU = .TRUE.
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    IF(PRNLEV >= 2) WRITE (OUTU,'(I6,A,I6,A,I6,A,A)') &
         NACTVE,' ATMS ',NACTC,' CLUS ',NACTG,' GRPS ', &
         'ARE ACTIVE '
    deallocate(flact)
    ! turn on activation flag (signifying active atoms have been selected)
    QNBACTON = .TRUE.
    RETURN
  END subroutine NBACTV

  SUBROUTINE MKCLUST(COMLYN,COMLEN)
    ! ----------------------------------------
    ! This routine sets up the generation of clusters,
    ! which are used in the BYCC non-bonded list generation
    ! routine (NBNDCC).
    ! The routine also calls ACTVDEF which sets the active
    ! cluster, group, and atom arrays to their default values.
    ! If the exclusions are to be regenerated (EXCL
    ! keyword), the MKCLUST calls SUEXTAB, which
    ! sets up and calls the MKEXTAB routine for creation
    ! of the exclusion table.
    ! MARGL is the cluster margin limit that is
    ! parsed and stored in the common block for
    ! use by NBNDCC.
    !               -RJPetrella 10.30.98
    !               modified 8.25.99 and 10.03 -RJP
    !
    ! -----------
    use chm_kinds
    use dimens_fcm
    use coord
    use psf
    use inbnd
    use bases_fcm
    use actclus_mod
    use stream
    use string
    use mexclar
    use number
    use contrl
    use memory
    use nbexcl,only:suextab

    implicit none

    CHARACTER COMLYN*(*)
    INTEGER COMLEN
    !
    real(chm_real) CADL
    LOGICAL QEXCLU
    !
!    INTEGER,parameter :: MXBPAT=6 ! maximum number of bonds per atom

    MARGL=GTRMF(COMLYN,COMLEN,'CMAR',FIVE)
    CADL=GTRMF(COMLYN,COMLEN,'CADL',FOUR)
    QEXCLU=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'EXCL') > 0) QEXCLU=.TRUE.
    !
    !    Initialize exclusions
    !
    ! decide what to do based on whether the exclusions are present
    ! and whether they need to be regenerated.
    !      IF (.NOT.QEXCTB) THEN  !if exclusion table not present
    !        call update, which will create exclusion and call mkclustdef
    !       IF(INBFRQ == 0) INBFRQ=999
    !       CALL UPDATE(COMLYN,0,X,Y,Z,WMAIN,.TRUE.,
    !     $        .FALSE.,.TRUE.,.FALSE.,.TRUE.,0,0,0,0,0,0,0)
    !      ELSE !if exclusion table is present
    ! set up and call the formation of exclusion table
    IF(QEXCLU)   & !if we are re-generating exclusion table
         CALL SUEXTAB(BNBND%IBLO14,BNBND%INB14, &
         BNBND%IGLO14,BNBND%ING14, &
         NATOM,NGRP)

    !
    !  call clustering routine
    !
!   print *,"calling clsbyrule"
    CALL CLSBYRULE(CADL)

    IF(.NOT.QNBACTON) CALL ACTVDEF !RJP 5.4.05
    CALL BOX27
    QCLSMD = .TRUE.
    RETURN
  END SUBROUTINE MKCLUST

  SUBROUTINE MKCLUSTDEF
    ! ----------------------------------------
    ! Same as MKCLUST except called by default
    ! at first update.
    ! Routine assumes exclusion lists have been
    ! made (for use in setting up exclusion tables).
    !               -RJP
    !
    ! -----------
  use chm_kinds
  use dimens_fcm
  use coord
  use psf
  use inbnd
  use bases_fcm
  use actclus_mod,only:margl,qclsmd,qnbacton
  use stream
  use mexclar
  use number
  use contrl
  use memory
    implicit none
    !
    real(chm_real) CADL
    !
!    INTEGER,parameter :: MXBPAT=6 ! maximum number of bonds per atom

    MARGL=FIVE
    CADL=FOUR
    !
    ! set up and call the formation of exclusion table
    !
    !  call clustering routine
    !
    CALL CLSBYRULE(CADL)

    IF(.NOT.QNBACTON) CALL ACTVDEF !RJP 5.4.05
    CALL BOX27
    QCLSMD = .TRUE.
    !
    RETURN
  END SUBROUTINE MKCLUSTDEF

  SUBROUTINE CLSBYRULE(CADL)
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  use actclus_mod
  use exfunc
  use coord
  use chutil,only:hydrog
  use memory
  use machutil,only:die
    implicit none
    !
    ! Creates 1,4 clusters from the list of bonds.
    ! Beginning from atom #1, the algorithm looks
    ! for the most highly populated cluster of atoms meeting
    ! the following criteria: 1) if N is the highest
    ! atom number contained in the cluster, all atoms
    ! from the first atom not yet clustered (i.e. atom 1 at
    ! the beginning) to atom N must be included.  That is,
    ! no atoms can be skipped.
    ! 2) No two atoms in the cluster can have a relationship
    ! that is more distant than 1,4.
    ! 3) there are no carbon or sulfur atoms at positions
    ! 1 or 4 in the cluster. This reduces the
    ! maximal spacial dimensions of the clusters.
    ! When the algorithm has found a cluster of atoms
    ! satisfying these criteria it stores the cluster and
    ! then repeats the procedure beginning with the first
    ! atom not yet clustered, and so on.
    ! For optimal non-bonded list generation efficiency (speed)
    ! clusters should be small in spatial dimension and few in
    ! number.  For a fixed number of clusters (and atoms),
    ! though, it is optimal to have a skewed population
    ! distribution; i.e. as many clusters as possible having
    ! only one or a few atoms and a small number of clusters
    ! having many atoms--hence the algorithm's attempt to
    ! select highly populated clusters.
    ! Although all atoms in an individual cluster have at
    ! most a 1,4 relationship in connectivity, the
    ! connecting atom or atoms between a given pair may
    ! be missing in that cluster (if, for example, the
    ! atom numbers are lower), so that a pair of atoms
    ! may appear "disconnected" within a given cluster.
    ! Individual atoms such as counter-ions are assigned
    ! their own 1-atom clusters.
    !
    ! The routine checks for gaps within segments
    ! by measuring the distance between consecutive
    ! atoms. If the distance exceeds a prespecified
    ! limit, the routine splits the cluster that
    ! contains the gap into two clusters. This helps
    ! prevent the creation of overly large clusters
    ! (which would slow down the non-bonded list
    ! generation performed in NBNDCC.)
    ! A single cluster can be split more than once.
    !
    !                                  -RJPetrella 8.25.99
    !
    !     passed variables
    real(chm_real) CADL

    !  work arrays
    INTEGER :: ATCLNM(NATOM+1)  !array of atom cluster numbers
    INTEGER :: BNDCNT(NATOM) ! number of bonds/atom
    INTEGER :: BNDBOT(NATOM) !base array for bond list
    INTEGER :: BNDLST(NATOM*MXBPAT) !bond list array
    INTEGER :: STORGE(NATOM) !flags (==1)atoms bonded to either of a tested pair
    LOGICAL :: NOHAT(NATOM)  !is the atom a N,O, or H
    INTEGER :: FRSTLST(MXBPAT) !list of atoms bound to first
    INTEGER :: SECLST(MXBPAT*MXBPAT+MXBPAT+1) !list of atoms bound to FRSTLST atoms or first
    INTEGER :: WKLIST(MXBPAT*MXBPAT+MXBPAT+1) !worklist of atoms forming possible clusters
    INTEGER :: TEMPCLS(2*MXBPAT+2) !temporary storage of cluster
    INTEGER :: IND1(2*MXBPAT+2)
    INTEGER :: IND2(2*MXBPAT+2)
    INTEGER :: CLLL(2*MXBPAT+2)

    !  local scalars
    INTEGER LASTCL  !number of the last cluster
    INTEGER BASE,LO,HI,I,J,PERM0,REMVD1,KK,BNDDAT
    INTEGER FIRST !first atom of a cluster
    INTEGER START !first atom of a cluster -1
    LOGICAL DONE,NOHITS,CONSEC,PAIRBD,NEWCLUS
    INTEGER ABSHICNT !highest number of atoms of any cluster
    INTEGER CTREM1 !number of atoms bound to first
    INTEGER CTREM2 !# of atoms in SECLST
    INTEGER CNTWK !# of atoms in WKLIST
    INTEGER HIGSTR,LOWSTR !hi and lo atoms in the storage array
    INTEGER TEMPCNT !number of atoms in TEMPCLS
    EXTERNAL EXCH5
    INTEGER AAA,BBB !first and second atomis in a tested I,J pair
    !
    ! Cluster-splitting variables
    INTEGER TT,SPLN,AT1,AT2
    INTEGER SCATCNT,JJ,K
    real(chm_real) XX,YY,ZZ,DIS
    INTEGER CN,CNM1,BEG,EN
    !
    INTEGER CCC
    INTEGER RNUM
    INTEGER IQ,IS,NAT
    integer :: CLLLMX
    !
    ! ----------------------------------
    ! MARGL -- common block variable representing
    !      the margin necessary to add to the cutoff
    !      distance in the non-bonded list generation
    !      routine to account for cluster size when
    !      setting up the cubical grid.
    !     (not used in this subroutine).
    !  XX,YY,ZZ --dummy work var's for dist calc
    !  DIS -- distance betw 2 consecutive atoms in a
    !      cluster
    !  CADL -- consecutive atom distance limit
    !      distance between consecutive atoms
    !    above which cluster is split
    ! SPLN  -- number of cuts, or splits, made on
    !       a given (overly large) cluster
    ! IND1,IND2 --work arrays that hold the cluster
    !    indices of atoms on either side of each split
    !    (i.e. ordinal value of where the atoms appear
    !     in the cluster--e.g. 2nd or 3rd).
    ! BEG,EN  variables storing the initial and final atom indices
    !     over which one must loop to create new atom
    !     clusters between two consecutive splits
    ! CLLL(CN) work array that stores atom numbers of the
    !     current cluster of (CN) atoms.
    ! SCATCNT  running count of all clustered atoms
    !
    !

    if(allocated(CLPRTN)) then
     call chmdealloc('nbndcc_util.src','CLSBYRULE','CLPRTN',natom,intg=CLPRTN)
     call chmdealloc('nbndcc_util.src','CLSBYRULE','CLHIGH',natom,intg=CLHIGH)
     call chmdealloc('nbndcc_util.src','CLSBYRULE','JCLUS',natom,intg=JCLUS)
     call chmdealloc('nbndcc_util.src','CLSBYRULE','CLUSNUM',natom,intg=CLUSNUM)
    endif

    call chmalloc('nbndcc_util.src','CLSBYRULE','CLPRTN',natom,intg=CLPRTN)
    call chmalloc('nbndcc_util.src','CLSBYRULE','CLHIGH',natom,intg=CLHIGH)
    call chmalloc('nbndcc_util.src','CLSBYRULE','JCLUS',natom,intg=JCLUS)
    call chmalloc('nbndcc_util.src','CLSBYRULE','CLUSNUM',natom,intg=CLUSNUM)
!
    IF(PRNLEV > 2) WRITE(OUTU,'(A)') 'Clustering atoms by rule'
    CLLLMX = 2*MXBPAT+2
    !
    LASTCL = 0
    ! initialize bondcount array and bondlist base array
    BNDCNT = 0
    BNDBOT = 0
    ! initialize atom cluster number array
    ATCLNM = 0
    DO I = 1,NATOM
       ! flag nitrogens, oxygens, and hydrogens
       IF(HYDROG(I).OR.((AMASS(I) < 16.1).AND.(AMASS(I) > 15.9)).OR. &
            ((AMASS(I) < 14.1).AND.(AMASS(I) > 13.9))) THEN
          NOHAT(I) = .TRUE.
       ELSE
          NOHAT(I) = .FALSE.
       ENDIF
    ENDDO
    !
    ! fill the bond counter array (#bonds for each atom)
    DO I = 1,NBOND
       BNDCNT(IB(I)) = BNDCNT(IB(I)) + 1
       BNDCNT(JB(I)) = BNDCNT(JB(I)) + 1
    ENDDO
    ! fill the bond list base array
    BASE = 0
    DO I = 1,NATOM
       BNDBOT(I) = BNDCNT(I) + BASE
       BASE = BNDBOT(I)
       BNDCNT(I) = 0
    ENDDO
    
    ! fill the bond list array
    DO I = 1,NBOND
       IF(IB(I) == 1) THEN
          BASE = 0
       ELSE
          BASE = BNDBOT(IB(I) -1)
       ENDIF
       BNDCNT(IB(I)) = BNDCNT(IB(I)) + 1
       BNDLST(BASE + BNDCNT(IB(I))) = JB(I)
       IF(JB(I) == 1) THEN
          BASE = 0
       ELSE
          BASE = BNDBOT(JB(I) -1)
       ENDIF
       BNDCNT(JB(I)) = BNDCNT(JB(I)) + 1
       BNDLST(BASE + BNDCNT(JB(I))) = IB(I)
    ENDDO
    ! *********************************************
    ! Start of main loop:
    !  Once through the loop = formation of 1 cluster
    ! *********************************************
    !     Initialize some work arrays
    DO I = 1,MXBPAT
       FRSTLST(I) = 0
    ENDDO
    DO I = 1,MXBPAT*MXBPAT+MXBPAT+1
        SECLST(I) = 0
        WKLIST(I) = 0
    ENDDO
    DO I = 1,NATOM
       STORGE(I) = 0
    ENDDO
    DO I = 1,2*MXBPAT+2
       TEMPCLS(I) = 0
    ENDDO
    START = 0  !the last atom clustered
    DONE = .FALSE.
    DO WHILE (.NOT. DONE)
       !  (NB: bond list is not sorted in psf)
       !  make a list of all atoms bonded to first atom
       !  then a list of all atoms bonded to those atoms
       ABSHICNT = 0
       CTREM1 = 0
       CTREM2 = 0
       FIRST = START + 1
       !  store the atoms bonded to first atom in a first
       ! list
       IF (FIRST  ==  1) THEN
          LO = 1
       ELSE
          LO = BNDBOT(FIRST - 1) + 1
       ENDIF
       HI = BNDBOT(FIRST)
       DO J = LO,HI
          CTREM1 = CTREM1 + 1
          FRSTLST(CTREM1) = BNDLST(J)
       ENDDO
       ! find the atoms bonded to any of those atoms
       ! and store in second list
       DO KK = 1,CTREM1
          REMVD1 = FRSTLST(KK)
          IF (REMVD1  ==  1) THEN
             LO = 1
          ELSE
             LO = BNDBOT(REMVD1 - 1) + 1
          ENDIF
          HI = BNDBOT(REMVD1)
          DO J = LO,HI
             CTREM2 = CTREM2 + 1
             SECLST(CTREM2) = BNDLST(J)
          ENDDO
          ! add atoms from first list to second list
          CTREM2 = CTREM2 + 1
          SECLST(CTREM2) = REMVD1
       ENDDO
       ! add the first atom to make sure it is included in the
       ! second list
       CTREM2 = CTREM2 + 1
       SECLST(CTREM2) = FIRST
       ! sort the second list
       CALL SORT(CTREM2,EXCH5,ORDER5,SECLST,0,0,0,0,0,0,1)
       ! eliminate double entries to create worklist
       IF (CTREM2 >= 1) THEN
          WKLIST(1) = SECLST(1)
       ENDIF
       CNTWK = 1
       DO I = 2,CTREM2
          IF (SECLST(I) /= SECLST(I-1)) THEN
             CNTWK = CNTWK + 1
             WKLIST(CNTWK) = SECLST(I)
          ENDIF
       ENDDO
       ! check all pairwise combinations of these atoms
       ! to see if they bond each other. IF a combination
       ! does, compile a list of the atoms bonded by either
       ! in the pair (plus the pair itself). Count all
       ! consecutive atoms from FIRST atom contained in this
       ! list of atoms.  This is the size of the cluster
       ! for the I,J pair.  The final cluster (before splits)
       ! is chosen to be the largest such cluster (i.e.
       ! containing the highest number of atoms) over all tested
       ! I,J pairs.
       !
       NOHITS = .TRUE.
       I = 0
       ! ________________________________________________
       ! loop through all pairwise combinations of atoms
       ! in the worklist
       ! ________________________________________________
       DO WHILE (I < (CNTWK-1))
          I = I + 1
          J = I
          DO WHILE (J < CNTWK)
             J = J + 1
             PAIRBD = .FALSE.
             !  NB: BBB should always be > AAA (from sort)
             AAA = WKLIST(I)   !1st of tested pair
             BBB = WKLIST(J)   !2nd of tested pair
             ! changed below RJP 1.12.00
             LOWSTR = 999999999
             HIGSTR = 0
             !  flag atoms bonding AAA in storage array
             !  (include AAA)
             IF (AAA  ==  1) THEN
                LO = 1
             ELSE
                LO = BNDBOT(AAA - 1) + 1
             ENDIF
             HI = BNDBOT(AAA)
             DO KK = LO,HI
                BNDDAT = BNDLST(KK)
                ! flag the atom if it is GE the FIRST
                ! atom, and it is either the other atom in the tested
                ! pair (carbons included) or a N,O, or H (carbons excluded).
                IF((BNDDAT >= FIRST).AND.(NOHAT(BNDDAT) &
                     .OR.(BNDDAT == BBB))) THEN
                   STORGE(BNDDAT) = 1
                   IF (BNDDAT < LOWSTR) LOWSTR = BNDDAT
                   IF (BNDDAT > HIGSTR) HIGSTR = BNDDAT
                   !  note if there is an intrapair bond
                ENDIF
                IF (BNDDAT == BBB) PAIRBD = .TRUE.
             ENDDO
             IF(AAA >= FIRST) THEN
                STORGE(AAA) = 1
                IF (AAA < LOWSTR) LOWSTR = AAA
                IF (AAA > HIGSTR) HIGSTR = AAA
             ENDIF
             !  flag atoms bonding BBB in storage array in same manner
             !  as above (include BBB)
             IF (BBB  ==  1) THEN
                LO = 1
             ELSE
                LO = BNDBOT(BBB - 1) + 1
             ENDIF
             HI = BNDBOT(BBB)
             DO KK = LO,HI
                BNDDAT = BNDLST(KK)
                IF((BNDDAT >= FIRST).AND.(NOHAT(BNDDAT) &
                     .OR.(BNDDAT == AAA))) THEN
                   STORGE(BNDDAT) = 1
                   IF (BNDDAT < LOWSTR) LOWSTR = BNDDAT
                   IF (BNDDAT > HIGSTR) HIGSTR = BNDDAT
                ENDIF
                IF (BNDDAT == AAA) PAIRBD = .TRUE.
             ENDDO
             IF(BBB >= FIRST) THEN
                STORGE(BBB) = 1
                IF (BBB < LOWSTR) LOWSTR = BBB
                IF (BBB > HIGSTR) HIGSTR = BBB
             ENDIF
             !  if there is an intrapair bond, continue
             IF (PAIRBD) THEN
                TEMPCNT = 0
                CONSEC = .TRUE. !consecutive atoms included in storge
                KK = FIRST
                ! count the number of consecutive atoms bonded to the
                !  inner pair (STORGE == 1) that are GE "FIRST"
                ! (and less than NATOM).
                DO WHILE (CONSEC.AND.(KK <= NATOM))
                   IF (STORGE(KK) == 1) THEN
                      TEMPCNT = TEMPCNT + 1
                      TEMPCLS(TEMPCNT) = KK
                      KK = KK + 1
                   ELSE
                      CONSEC = .FALSE.
                   ENDIF
                ENDDO
                ! if the cluster is larger than all previous
                ! for this I,J pair, save high atom number
                IF (TEMPCNT > ABSHICNT) THEN
                   NOHITS = .FALSE.
                   ABSHICNT = TEMPCNT
                ENDIF
                DO KK = 1,TEMPCNT
                   TEMPCLS(KK) = 0
                ENDDO
             ENDIF !IF pairbd or not
             !  clear storage array
             DO KK = LOWSTR,HIGSTR
                STORGE(KK) = 0
             ENDDO
          ENDDO !J loop
       ENDDO !I loop
       ! _________________________________________________
       ! End of loop over pairwise combinations of atoms
       ! in worklist
       ! _________________________________________________
       IF (CNTWK > 1) THEN  !multiple-atom cluster
          IF (.NOT.NOHITS) THEN
             LASTCL = LASTCL + 1
             DO KK = 1,ABSHICNT
                ATCLNM(START + KK) = LASTCL
             ENDDO
             START = START + ABSHICNT
          ELSE !No hits
             CALL WRNDIE(-5,'<CLSBYRULE>','NO POSSIBLE CLUSTERS')
          ENDIF
       ELSE !CNTWK = 1 (single-atom cluster)
          LASTCL = LASTCL + 1
          ATCLNM(FIRST) = LASTCL
          START = FIRST
       ENDIF
       IF (START == NATOM) DONE = .TRUE.
    ENDDO !end main loop
    ! **************************************************
    ! **************************************************
    !  cluster splitting:
    !  loop through each preliminary cluster, noting
    !  when you are starting a new cluster
    SPLN = 0
    SCATCNT = 0
    CLNUM = 0
    CN = 0
    NEWCLUS = .FALSE.
    I = 1
    CCC = ATCLNM(1)
    DO WHILE(I <= NATOM)
       CN = 0
       DO WHILE ((.NOT.NEWCLUS).AND.(I <= NATOM))
          CN = CN + 1
          CLLL(CN) = I
          I = I + 1
          IF (ATCLNM(I) /= CCC) THEN
             NEWCLUS = .TRUE.
             CCC = ATCLNM(I)
          ENDIF
       ENDDO
       !  If new cluster, check consecutive atom distances
       SPLN = 0
       DO J = 1,CN -1
          AT1 = CLLL(J)
          K = J + 1
          AT2 = CLLL(K)
          XX = X(AT2) - X(AT1)
          XX = XX*XX
          YY = Y(AT2) - Y(AT1)
          YY = YY*YY
          ZZ = Z(AT2) - Z(AT1)
          ZZ = ZZ*ZZ
          DIS = XX + YY + ZZ
          !  IF distance too great, store this position
          IF (DIS  > (CADL*CADL)) THEN
             !            IF (PRNLEV > 2) THEN
             !      WRITE(OUTU,100) 'SPLITTING CLUSTERS BETWEEN ATMS',
             !     & AT1,' & ',AT2,' SQ DISTANCE= ',DIS
             ! 100     FORMAT(A31,I6,A3,I6,A11,F7.2)
             !            ENDIF
             SPLN = SPLN + 1
             IND1(SPLN) = J
             IND2(SPLN) = K
          ENDIF
       ENDDO
       !  Perform all required splits, if any.
       !  Store the smaller clusters in the
       !  final linked cluster lists
       IF (SPLN > 0) THEN
          IND1(SPLN + 1) = CN
          DO JJ = 1,SPLN + 1
             IF (JJ == 1) THEN
                BEG = 1
             ELSE
                BEG =IND2(JJ-1)
             ENDIF
             EN = IND1(JJ)
             CLNUM = CLNUM + 1
             CLPRTN(CLNUM) = EN -BEG + 1
             DO TT = BEG,EN
                SCATCNT = SCATCNT + 1
                JCLUS(SCATCNT) = CLLL(TT)
             ENDDO
             CLHIGH(CLNUM) = SCATCNT
          ENDDO
       ELSE IF (CN > 0) THEN
          CLNUM = CLNUM + 1
          CLPRTN(CLNUM) = CN
          DO JJ=1,CN
             SCATCNT = SCATCNT + 1
             JCLUS(SCATCNT) = CLLL(JJ)
          ENDDO
          CLHIGH(CLNUM) = SCATCNT
       ENDIF
       DO J = 1,CLLLMX
          CLLL(J) = 0
       ENDDO
       CN = 0
       NEWCLUS = .FALSE.
    ENDDO
    !      IF(PRNLEV >= 2) WRITE (OUTU, '(A,I7)')
    !     $' total  # CLUSTERS made = ',CLNUM
    !
    !   create an inverse mapping array containing
    !  the cluster number for each atom
    DO I=1,CLNUM
       IQ = CLHIGH(I)
       IS = IQ - CLPRTN(I) + 1
       NAT=CLPRTN(I)
       IF(NAT <= 0) CALL DIE
       !         CNT = 0
       DO TT = IS,IQ
          J = JCLUS(TT)
          CLUSNUM(J) = I
          !          CNT = CNT + 1
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE CLSBYRULE

  SUBROUTINE ACTVDEF
    !  Sets the active atom array
    !  equal to the array of all atoms in system.
    !  Sets active cluster array to all clusters.
    !  Sets active groups to all groups.
    !       -RJ Petrella 1.6.99
  use chm_kinds
  use dimens_fcm
  use psf
  use actclus_mod
  use stream
  use memory

    implicit none
    !
    !. Local var
    INTEGER I
! these arrays are always allocated and deallocated together
    if(allocated(ACTV)) then
     call chmdealloc('nbndcc_util','ACTVDEF','ACTV',NATOM,intg=ACTV)
     call chmdealloc('nbndcc_util','ACTVDEF','ACAFLG',NATOM,intg=ACAFLG)
     call chmdealloc('nbndcc_util','ACTVDEF','JACLS',NATOM,intg=JACLS)
     call chmdealloc('nbndcc_util','ACTVDEF','ACLPTN',CLNUM,intg=ACLPTN)
     call chmdealloc('nbndcc_util','ACTVDEF','ACLHI',CLNUM,intg=ACLHI)
     call chmdealloc('nbndcc_util','ACTVDEF','ACTVG',NGRP,intg=ACTVG)
     call chmdealloc('nbndcc_util','ACTVDEF','ACGFLG',NGRP,intg=ACGFLG)
    endif

    call chmalloc('nbndcc_util','ACTVDEF','ACTV',NATOM,intg=ACTV)
    call chmalloc('nbndcc_util','ACTVDEF','ACAFLG',NATOM,intg=ACAFLG)
    call chmalloc('nbndcc_util','ACTVDEF','JACLS',NATOM,intg=JACLS)
    call chmalloc('nbndcc_util','ACTVDEF','ACLPTN',CLNUM,intg=ACLPTN)
    call chmalloc('nbndcc_util','ACTVDEF','ACLHI',CLNUM,intg=ACLHI)
    call chmalloc('nbndcc_util','ACTVDEF','ACTVG',NGRP,intg=ACTVG)
    call chmalloc('nbndcc_util','ACTVDEF','ACGFLG',NGRP,intg=ACGFLG)
    !
    DO I = 1,NATOM
       ACTV(I) = I
       ACAFLG(I) = 1
    ENDDO
    NACTVE = NATOM
    !
    NACTC = CLNUM
    !
    DO I=1,CLNUM
       ACLPTN(I) = CLPRTN(I)
       ACLHI(I) = CLHIGH(I)
    ENDDO
    DO I = 1,NATOM
       JACLS(I) = JCLUS(I)
    ENDDO
    !
    DO I = 1,NGRP
       ACTVG(I) = I
       !  Added below --RJP
       ACGFLG(I) = 1
    ENDDO
    NACTG = NGRP
    IF(PRNLEV >= 2) WRITE (OUTU,'(I6,A,I6,A,I6,A,A)') &
         NACTVE,' ATMS ',NACTC,' CLUS ',NACTG,' GRPS ', &
         'ARE ACTIVE '
    !
    QNBACTON = .TRUE.
    RETURN
  END SUBROUTINE ACTVDEF
end module nbndcc_util

