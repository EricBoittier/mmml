module bycc_mod

  use dimens_fcm
  use chm_kinds
  use actclus_mod
  implicit none

contains

SUBROUTINE NBNDCCF(NNNB,JNB,MAXJNB, &
     INBLO,X,Y,Z, &
     INB14,IBLO14, &
     CUTNB,CPNER,CPDIS,NCPNER,NCPDIS, &
     CMPLTD,PNTNER,PNTDIS,PARTNN, &
     PARTND,ORDCBN,XOO,YOO,ZOO, &
     DUMMYX,HAFMAR,MEANX,MEANY,MEANZ, &
     LSTN0,NUMMP, &
     HIMPRTN,WRKAR,MMM,LSTM0, &
     CNTN,XO,YO,ZO,MXNCUB, &
     EXCLAR,ATEXPT,NUEXCA, &
     ATSX,ATSY,ATSZ)
#if KEY_NO_BYCC==0 /*yesbycc*/
  !
  !-----------------------------------------------------------------------
  !  Made from NBNDGC and NBONDG in late 1998 by R.J.Petrella.
  !  The by-clusters-in-cubes, or BYCC, algorithm is a combination
  !  of the BYCUbes algorithm, written by J. Thomas Ngo (June, 1990),
  !  and the BYGRoups algorithm, written by Bernard R. Brooks
  !  (Aug 22, 1981). BYCC first places the system in a cubical grid,
  !  as BYCU does, and then does distance comparisons on clusters
  !  of atoms, as BYGR essentially does, rather than on individual atoms.
  !  Both steps allow for a reduction in the number of atoms passed to
  !  the final interatomic distance loop, speeding the calculation.
  !  The use of both the cubical partitioning and atomic grouping
  !  techniques in BYCC allows for greater efficiency than is possible
  !  using either technique alone, and hence BYCC is generally faster
  !  than either BYCU or BYGR. Also, because the routine does the final
  !  atom-atom distance tests, the exclusions, and the formation of the
  !  non-bonded list all in one final loop, only a cluster-cluster
  !  pairwise list needs to be stored internally to the routine
  !  as a work array. Thus, the memory requirements are reduced
  !  relative to BYCU, which stores a much longer atom-atom
  !  pairwise list (essentially a second non-bonded list) internally.
  !
  !  Like BYCU, the computational time for BYCC increases linearly
  !  with the number of atoms and in a sigmoidal fashion with
  !  cut-off distance.  At (cut-off distance) << (radius of system)
  !  the dependence on cut-off distance is essentially third order
  !  but depends on the system and the actual cutoff value.
  !
  !  An additional feature of this algorithm is its ability to handle
  !  "active" atom selections.  This allows for focusing of calcula-
  !  tions on regions of interest, by ignoring atoms that are defined
  !  as "inactive." The routine loops over only the active clusters,
  !  which is the set of clusters that contain at least one active
  !  atom each. Inactive atoms should have been eliminated from the
  !  active clusters prior to having been passed to BYCC.
  !
  !  ASSUMPTIONS:
  !  BYCC assumes that the atomic cluster numbers are in order,
  !  such that for any two cluster numbers, I and J, if J > I,
  !  then all atoms contained in cluster J have higher atom numbers
  !  than all the atoms contained in cluster I.
  !  The order of the atoms as they appear in the clusters must also
  !  be the same as their order as defined by the RTF/PSF.
  !  BYCC assumes the existence of the exclusion table (exclusions
  !  in tabular format).
  !  It does not warn about too-close atoms.
  !  BYCC no longer assumes any two atoms in a given cluster are closer
  !  than the non-bonded cutoff distance (i.e. clusters larger than
  !  the cutoff distance are allowed).
  !
  !  Many thanks to B.R. Brooks and J.T. Ngo.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
!!!   use actclus_mod
  use contrl
  use comand
  use parallel
  use corman3,only:squareup
  use machutil,only:die
  !
  implicit none
  ! *****************************************************************************
  ! PASSED VARIABLES
  INTEGER NNNB,MAXJNB,NCPNER,NCPDIS
  INTEGER JNB(*),INBLO(*)
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER INB14(*),IBLO14(*)
  real(chm_real) CUTNB
  LOGICAL CMPLTD
  INTEGER MXNCUB ! maximum number of cubes in system
  INTEGER EXCLAR(*),ATEXPT(*),NUEXCA(*)  !exclusion table
  real(chm_real)  ATSX(*),ATSY(*),ATSZ(*) !for heuristic update
  !
  ! *****************************************************************************
  ! "LOCAL" ARRAYS PASSED FOR MEMORY ALLOCATION ONLY
  !
  INTEGER CPNER(*),CPDIS(*) !2nd-cluster lists (distant & near)
  INTEGER PNTNER(*),PNTDIS(*) ! pointers into CPNER and CPDIS
  INTEGER PARTND(*) !number of distant partners for a cluster
  INTEGER PARTNN(*) !number of nearby partners for a cluster
  INTEGER ORDCBN(*) ! ordinal number cube position of cluster
  INTEGER XOO(*),YOO(*),ZOO(*) !cubical grid coord of cluster
  INTEGER DUMMYX(*) !dummy storage array
  real(chm_real) HAFMAR(*) !largest center-to-atom distance in a cluster
  real(chm_real) MEANX(*),MEANY(*),MEANZ(*) !Cartes coord of cluster
  INTEGER LSTN0(*) !atom of next-highest number in the same cube
  !  Cube arrays
  INTEGER NUMMP(*)    ! # of non-empty partner-cubes in surroundings
  INTEGER HIMPRTN(*)  ! pointer into partner-cube list (WRKAR)
  INTEGER WRKAR(*) ! partner-cube list
  INTEGER MMM(*)      ! ordinal number of each non-empty cube on grid
  INTEGER LSTM0(*)    ! highest atom number contained in a cube
  INTEGER CNTN(*)     ! # of clusters contained in a cube
  INTEGER XO(*),YO(*),ZO(*) !cubical grid coord of cubes
  ! ****************************************************************************
  ! LOCAL SCALARS
  !  for exclusions
  INTEGER DIFF ! diffrnce between atom numbers in excluded pairs
  INTEGER MAXDIF !max diffrnce between atom numbers in excluded pairs
  INTEGER EXCL14 !exclusion flag (if==0, atom pair is excluded)
  INTEGER ELEN !table index
  !  for cubes
  real(chm_real) CUBL,CUBNEED !cube lngth; necessary cube lngth for v. larg system
  INTEGER TOTCUB  !total number of cubes in system
  INTEGER FLDM !total number of non-empty cubes
  INTEGER NUMBX,NUMBY,NUMBZ ! dimensions of system in cube-lengths
  INTEGER NXY               ! dimension of system in XY plane
  real(chm_real) MAXX(3),MINX(3),MAXY(3),MINY(3),MAXZ(3),MINZ(3)
  INTEGER MNUM !size of cube-cube partner list (WRKAR)
  INTEGER PP,ZER ! temp storage of ordinal cube number
  !  for clusters
  real(chm_real) DXT,DYT,DZT !sum of atm coordinates for avrgs in clusters
  real(chm_real) DXN,DYN,DZN !distnce components of atoms from center of clusters
  real(chm_real) XAVG,YAVG,ZAVG  ! ave position of all clusters
  real(chm_real) XSUM,YSUM,ZSUM ! summed coordinates for averages over all clusters
  real(chm_real) MARG !cluster margin (distance added to cutnb for testing)
  real(chm_real) OMARG
  real(chm_real) IMAR
  real(chm_real) IMARSQ !interclus dist below which interatomic dist MUST be <CUTNB
  real(chm_real) MARGSQ !interclus dist below which interatomic dist possibly<CUTNB
  real(chm_real) DMAX1,DMAX2 !1st and 2nd highest half-margin distances
  !  for atoms
  real(chm_real) DSQ ! temp storage of squared distance
  real(chm_real) XX1,YY1,ZZ1,X1,Y1,Z1  !temp coordinate storage
  real(chm_real) CTNBSQ ! square of cutoff distance
  LOGICAL FIXED  !fixed atom (IMOVE  >  0)
  !  various indices and other
  real(chm_real) GRMAX,STOR,XR,YR,ZR !temporary storage variables
  real(chm_real) MAC1X,MAC1Y,MAC1Z,XM,YM,ZM !temporary coordnate storage
  INTEGER B1,B2,B3,R,STRTAT
  INTEGER ACG,IS2,IQ2,ACT,AT1,AT2,ACC,AC1,AC2
  INTEGER ACC1,AC,BB,DD
  INTEGER IS,IQ,NAT,I,N,J,AT,T,KK,M,CC,TT,UU,SS
  INTEGER PLVOLD
  INTEGER ITEMP,NPR
  !
  ! *****************************************************************************
  ! ************* End of variables for BYCC *************************************
  ! *****************************************************************************
  !
  PLVOLD = PRNLEV
  IF (DYNAMQ.OR.MINXYZ) PRNLEV = 3
  !
  IF(PRNLEV > 3) WRITE(OUTU,'(A)') 'Using C-IN-C search'
  !     Store atom positions
  DO I=1,NATOM
     ATSX(I)=X(I)
     ATSY(I)=Y(I)
     ATSZ(I)=Z(I)
  ENDDO
  CTNBSQ = CUTNB*CUTNB
  !C
  !-- Calculate the average positions of clusters and the -----------------
  ! -- cluster dimensions and cluster margin necessary   ------------------
  DMAX1 = 0
  DMAX2 = 0
  ! --added error message below RJP
  IF (NACTC == 0) CALL WRNDIE(-5,'<NBNDCCF>', &
       'NO ACTIVE CLUSTERS DEFINED')
  DO I = 1,NACTC
     ! ------------------------------------
     ! initialize some cluster-based arrays
     CPNER(I) = 0
     CPDIS(I) = 0
     PNTNER(I) = 0
     PNTDIS(I) = 0
     PARTND(I) = 0
     PARTNN(I) = 0
     ORDCBN(I) = 0
     XOO(I) = 0
     YOO(I) = 0
     ZOO(I) = 0
     DUMMYX(I) = 0
     HAFMAR(I) = 0
     MEANX(I) = 0
     MEANY(I) = 0
     MEANZ(I) = 0
     ! -------------------------------------
     IQ= ACLHI(I)
     IS=IQ - ACLPTN(I) + 1
     NAT = ACLPTN(I)
     IF (NAT <= 0) CALL DIE
     DXT=0
     DYT=0
     DZT=0
     DO BB = IS,IQ
        J = JACLS(BB)
        DXT=DXT+X(J)
        DYT=DYT+Y(J)
        DZT=DZT+Z(J)
     END DO
     MEANX(I)=DXT/NAT
     MEANY(I)=DYT/NAT
     MEANZ(I)=DZT/NAT
     GRMAX = 0
     DO BB = IS,IQ
        J = JACLS(BB)
        DXN=X(J) - MEANX(I)
        DXN = DXN*DXN
        DYN=Y(J) - MEANY(I)
        DYN = DYN*DYN
        DZN=Z(J) - MEANZ(I)
        DZN = DZN*DZN
        DSQ = DXN+DYN+DZN
        STOR = SQRT(DSQ)
        IF (STOR > GRMAX) THEN
           GRMAX = STOR
        ENDIF
        IF(DSQ > DMAX1) THEN
           DMAX2 = DMAX1
           DMAX1 = DSQ
        ELSE IF (DSQ > DMAX2) THEN
           DMAX2 = DSQ
        ENDIF
     ENDDO  !over BB
     HAFMAR(I) = GRMAX
  ENDDO
  MARG = SQRT(DMAX2) + SQRT(DMAX1)
  IF (PRNLEV > 3) THEN
     WRITE (OUTU,'(A,F6.4)') 'CALCed CLUS MARGIN = ', &
          MARG
  ENDIF
  IF (MARG > 5) THEN
     CALL WRNDIE(1,'<NBNDCCF>', &
          'LARGE CLUSTERS PRESENT, MARGIN > 5A')
  ENDIF
  IF (MARG > MARGL) THEN
     CALL WRNDIE(1,'<NBNDCCF>', &
          'CLUSTER MARGINS EXCEED SPECIFIED LIMIT')
     IF (PRNLEV >= 2) THEN
        WRITE(OUTU,*) '***************************'
        WRITE(OUTU,*) 'WARNING: CLUSTER MARGINS EXCEED '
        WRITE(OUTU,*) 'SPECIFIED LIMIT'
        WRITE(OUTU,*) 'CLUS MARGIN SET TO LIMIT = ', &
             MARGL
        WRITE(OUTU,*) 'SOME ATOM PAIRS MAY BE LOST'
        WRITE(OUTU,*) '***************************'
        WRITE(OUTU,*) ' '
     ENDIF
     MARG = MARGL
  ENDIF
2 FORMAT(A,F8.2,A,F8.2,A,F8.2)
3 FORMAT(A,I6,A,F8.2)
4 FORMAT(A,F5.3)
  !
  CMPLTD=.FALSE.            ! Will remain .FALSE. until very end
  !
  !--- Initialize the non-bonded list --necessary since
  ! unassigned elements MUST equal zero later in algorithm.
  DO I = 1,NATOM
     INBLO(I) = 0
  ENDDO
  !
  ! --set cube length equal to (cut-off dist + margin width ) --
  CUBL = CUTNB + MARG
  IF (PRNLEV > 3) WRITE (OUTU,'(A,F6.3)') &
       'CUBE LENGTH ',CUBL
  ! --initialize sums for center of geometry calculation
  ! initialize coordinate limits -------------------------------
  XSUM = 0
  YSUM = 0
  ZSUM = 0
  DO KK = 1,3
     MINX(KK) = 99999
     MAXX(KK) = -99999
     MINY(KK) = 99999
     MAXY(KK) = -99999
     MINZ(KK) = 99999
     MAXZ(KK) = -99999
  ENDDO
  ! -------- Loop over clusters to find center of geometry
  ! and coordinate limits (in terms of clusters)
  ! -------------------------------------------------------
  DO ACC = 1,NACTC
     XSUM = XSUM + MEANX(ACC)
     YSUM = YSUM + MEANY(ACC)
     ZSUM = ZSUM + MEANZ(ACC)
     IF (MEANX(ACC) > MAXX(1)) THEN
        MAXX(1) = MEANX(ACC)
        MAXX(2) = MEANY(ACC)
        MAXX(3) = MEANZ(ACC)
     ENDIF
     IF (MEANX(ACC) < MINX(1)) THEN
        MINX(1) = MEANX(ACC)
        MINX(2) = MEANY(ACC)
        MINX(3) = MEANZ(ACC)
     ENDIF
     IF (MEANY(ACC) > MAXY(2)) THEN
        MAXY(1) = MEANX(ACC)
        MAXY(2) = MEANY(ACC)
        MAXY(3) = MEANZ(ACC)
     ENDIF
     IF (MEANY(ACC) < MINY(2)) THEN
        MINY(1) = MEANX(ACC)
        MINY(2) = MEANY(ACC)
        MINY(3) = MEANZ(ACC)
     ENDIF
     IF (MEANZ(ACC) > MAXZ(3)) THEN
        MAXZ(1) = MEANX(ACC)
        MAXZ(2) = MEANY(ACC)
        MAXZ(3) = MEANZ(ACC)
     ENDIF
     IF (MEANZ(ACC) < MINZ(3)) THEN
        MINZ(1) = MEANX(ACC)
        MINZ(2) = MEANY(ACC)
        MINZ(3) = MEANZ(ACC)
     ENDIF
  ENDDO
  XAVG = XSUM/NACTC
  YAVG = YSUM/NACTC
  ZAVG = ZSUM/NACTC
5 FORMAT(A,F8.2,A,F8.2)
  DO KK = 1,3
     IF ((MAXX(KK) >= 9999).OR.(MAXY(KK).GE.9999).OR. &
          (MAXZ(KK) >= 9999)) THEN
        CALL WRNDIE(-5,'<NBNDCCF>', &
             'NBNDCCF> SOME COORDNTS >= 9999 (UNDEFINED)')
     ENDIF
  ENDDO
  NUMBX = INT((MAXX(1)-MINX(1))/CUBL) + 1
  NUMBY = INT((MAXY(2)-MINY(2))/CUBL) + 1
  NUMBZ = INT((MAXZ(3)-MINZ(3))/CUBL) + 1
  TOTCUB = NUMBX*NUMBY*NUMBZ
  ! If cubical grid is too large, try to reorient molecule -------
  IF (TOTCUB > MXNCUB) THEN
     IF (PRNLEV > 3) THEN
        WRITE(OUTU,'(A,I8)') 'TOTAL CUBES = ', TOTCUB, &
             'EXCEED LIM = ', MXNCUB
        WRITE(OUTU,'(A)') 'TRYING TO SQUARE UP STRUCTURE'
     ENDIF
14   FORMAT(A,I10)
15   FORMAT(A,F8.2)
     DO I = 1,NACTC
        DUMMYX(I) = I
     ENDDO
     !
     CALL SQUAREUP(MAXX,MINX,MAXY,MINY,MAXZ,MINZ,XAVG, &
          YAVG,ZAVG,MEANX,MEANY,MEANZ,NACTC,DUMMYX,NACTC)
     !
     XR = MAXX(1)-MINX(1)
     YR = MAXY(2)-MINY(2)
     ZR = MAXZ(3)-MINZ(3)
     NUMBX = INT(XR/CUBL) + 1
     NUMBY = INT(YR/CUBL) + 1
     NUMBZ = INT(ZR/CUBL) + 1
     TOTCUB = NUMBX*NUMBY*NUMBZ
     IF (PRNLEV > 3) THEN
        WRITE(OUTU,14) 'TOTOL CUBES (AFTER SQUARE) = ', &
             TOTCUB
     ENDIF
     ! -- IF grid is still too large, adjust cube length ----------------
     IF (TOTCUB > MXNCUB) THEN
        CUBNEED = CUBL
        DO WHILE(TOTCUB > MXNCUB)
           CUBNEED = ((XR + CUBNEED)*(YR + CUBNEED)* &
                (ZR + CUBNEED))
           CUBNEED = ((CUBNEED/MXNCUB)**0.33333333)
           NUMBX = INT(XR/CUBNEED) + 1
           NUMBY = INT(YR/CUBNEED) + 1
           NUMBZ = INT(ZR/CUBNEED) + 1
           TOTCUB = NUMBX*NUMBY*NUMBZ
           WRITE(OUTU,'(A,F6.3,A,I9)') 'CNEED ',CUBNEED, &
                ' TOTCUB ',TOTCUB
        ENDDO
        CUBL = CUBNEED
        IF (PRNLEV >= 2) THEN
           WRITE(OUTU,15) 'EXPANDING CUBL to CUBNEED= ', &
                CUBL
        ENDIF
     ENDIF ! overall grid still large after reorient
  ENDIF ! overall grid too large
  !
  IF (PRNLEV > 3) THEN
     WRITE (OUTU,'(A)') &
          'NBNDCCF Building particle interaction list using grid'
     WRITE (OUTU,'(A,I8)')' Number of active atoms         =', NACTVE
     WRITE (OUTU,'(A,I8)')' Number of active clusters      =', NACTC
     WRITE (OUTU,'(A,I8)')' Number of cells in X dimension =', NUMBX
     WRITE (OUTU,'(A,I8)')' Number of cells in Y dimension =', NUMBY
     WRITE (OUTU,'(A,I8)')' Number of cells in Z dimension =', NUMBZ
     WRITE (OUTU,'(A,I8)')' Number of cells, total         =', TOTCUB
  ENDIF
  IF (PRNLEV > 3) WRITE (OUTU,'(A,F7.2)') &
       ' Cell size                      =', CUBL
  !
  !
  NXY = NUMBX*NUMBY
  ! ---initialize some cube arrays ------------------------------
  DO I = 1, TOTCUB
     LSTM0(I) = 0
     CNTN(I) = 0
     NUMMP(I) = 0
     HIMPRTN(I) = 0
     MMM(I) = 0
     XO(I) = 0
     YO(I) = 0
     ZO(I) = 0
  ENDDO
  ! -------------------------------------------------------------
  ! check to see which cubes are "filled" (non-empty) and
  ! for the filled ones, store:
  ! their positions on the cubical grid (XO,YO,ZO)
  ! and the number of clusters they contain (CNTN)
  ! Also, store the total number of filled cubes (FLDM)
  FLDM = 0
  DO AC = 1,NACTC
     LSTN0(AC) = 0
     PARTNN(AC) = 0
     PARTND(AC) = 0
     XOO(AC) = INT((MEANX(AC)-MINX(1))/CUBL)
     YOO(AC) = INT((MEANY(AC)-MINY(2))/CUBL)
     ZOO(AC) = INT((MEANZ(AC)-MINZ(3))/CUBL)
     ORDCBN(AC) = XOO(AC) + (YOO(AC))*NUMBX + &
          (ZOO(AC))*NXY + 1
     IF (ORDCBN(AC) > MXNCUB) THEN
        CALL WRNDIE(-5,'<NBNDCCF>', &
             'NBNDCCF> TOO MANY CUBES. INCRSE CUBL OR REDUCE SYSTEM')
     ENDIF
     M = ORDCBN(AC)
     IF (CNTN(M) == 0) THEN
        FLDM = FLDM + 1
        MMM(FLDM) = M
        !  Note:  MMM is NOT sorted
        ZO(M) = ZOO(AC)
        YO(M) = YOO(AC)
        XO(M) = XOO(AC)
     ENDIF
     CNTN(M) = CNTN(M) + 1
  ENDDO
  ! --------------------------------------------------------------
  ! From Tom Ngo's "intertwined lists" algorithm.
  ! For each cube, the highest-numbered cluster contained in the
  ! cube is stored (in LSTM0). This cluster is also linked to the
  ! next highest cluster in number contained in the cube, such that
  ! LSTN0(high) = next highest. This is done for all clusters in
  ! a given cube. Thus the clusters within a particular cube are
  ! linked to each other in a "chain" and they are all linked
  ! to the cube number via the high cluster number.
  DO ACC1 = 1, NACTC
     M = ORDCBN(ACC1)
     LSTN0(ACC1) = LSTM0(M)
     LSTM0(M) = ACC1
  ENDDO
  ! -------------------------------------------------------------
  ! For each filled cube, determine the set of filled cubes in its
  ! immediate surroundings and store the ordinal-number positions
  ! of these filled cubes in a linked list.
  MNUM = 0
  DO CC = 1,FLDM
     ZER = MMM(CC)
     KK = 1
     DO WHILE(KK <= 27)
        IF(((BBX(KK)+XO(ZER)) >= 0).AND.((BBY(KK)+YO(ZER)).GE.0).AND. &
             ((BBZ(KK)+ZO(ZER)) >= 0)) THEN
           !  Next line means the cube can't be out of bounds
           IF(((BBX(KK)+XO(ZER)) < NUMBX).AND. &
                ((BBY(KK)+YO(ZER)) < NUMBY) &
                .AND.((BBZ(KK)+ZO(ZER)) < NUMBZ)) THEN
              PP = BBX(KK) + BBY(KK)*NUMBX + BBZ(KK)*NXY + ZER
              IF (CNTN(PP) > 0) THEN
                 MNUM = MNUM + 1
                 NUMMP(ZER) = NUMMP(ZER) + 1
                 WRKAR(MNUM) = PP
              ENDIF
           ENDIF
        ENDIF
        KK = KK + 1
     ENDDO
     HIMPRTN(ZER) = MNUM
  ENDDO
  !
  !  ----------------------------------------------------------------
  !  For a given "test" cluster, loop over the set of surrounding (filled)
  !  cubes, and for each cube, compare the position of each cluster
  !  within the cube to that of the test cluster.  Save a cluster pair
  !  only if the intercluster distance is less than (the non-
  !  bonded cutoff + the "half-margins" of both clusters). Exclude a
  !  cluster pair if the second cluster number <= the first.
  NCPNER = 0
  NCPDIS = 0
#if KEY_PARALLEL==1
  DO AC1 = MYNODP,NACTC,NUMNOD
#else /**/
  DO AC1 = 1,NACTC
#endif 
     ZER = ORDCBN(AC1)
     PARTNN(AC1) = 0
     PARTND(AC1) = 0
     MAC1X = MEANX(AC1)
     MAC1Y = MEANY(AC1)
     MAC1Z = MEANZ(AC1)
     UU = HIMPRTN(ZER)
     SS = UU - NUMMP(ZER) + 1
     DO WHILE(SS <= UU)
        M = WRKAR(SS)
        J = 1
        AC2 = LSTM0(M)
        DO WHILE ((J <= CNTN(M)).AND.(AC2 > AC1))
           IMAR = CUTNB - HAFMAR(AC2) - HAFMAR(AC1)
           IF(IMAR < 0) THEN
              IMARSQ = 0
           ELSE
              IMARSQ = IMAR*IMAR
           ENDIF
           OMARG = (HAFMAR(AC2) + HAFMAR(AC1) &
                + CUTNB)
           MARGSQ = OMARG*OMARG
           XM = MEANX(AC2) - MAC1X
           YM = MEANY(AC2) - MAC1Y
           ZM = MEANZ(AC2) - MAC1Z
           DSQ = XM*XM+YM*YM+ZM*ZM
           IF (DSQ  <=  IMARSQ) THEN
              NCPNER = NCPNER + 1
              CPNER(NCPNER) = AC2
              PARTNN(AC1) = PARTNN(AC1) + 1
           ELSE IF (DSQ  <=  MARGSQ) THEN
              NCPDIS = NCPDIS + 1
              CPDIS(NCPDIS) = AC2
              PARTND(AC1) = PARTND(AC1) + 1
           ENDIF
           AC2 = LSTN0(AC2)
           J = J + 1
        ENDDO
        SS = SS +1
     ENDDO
     IF (((NCPDIS + NACTC) > MAXACD).OR. &
          ((NCPNER + NACTC) > MAXACN)) THEN
        IF ((NCPDIS + NACTC) > MAXACD) THEN
           NCPDIS = MAXACD + 1
#if KEY_PARALLEL==1 /*maxacdsze*/
           IF(NUMNOD > 1) THEN
              WRITE(OUTU,'(A)') &
                   ' DISTANT CLUSTER ARRAY SIZE EXCEEDED; INCREASE NBSCALE'
              CALL WRNDIE(-5,'<NBNDCCO>','ARRAY SIZE LIMIT EXCEEDED')
           ENDIF
#endif /* (maxacdsze)*/
           RETURN
        ENDIF
        IF ((NCPNER + NACTC) > MAXACN) THEN
           NCPNER = MAXACN + 1
#if KEY_PARALLEL==1 /*maxacnsze*/
           IF(NUMNOD > 1) THEN
              WRITE(OUTU,'(A)') &
                   ' NEAR CLUSTER ARRAY SIZE EXCEEDED; INCREASE NBSCALE'
              CALL WRNDIE(-5,'<NBNDCCO>','ARRAY SIZE LIMIT EXCEEDED')
           ENDIF
#endif /* (maxacnsze)*/
           RETURN
        ENDIF
     ENDIF
     PNTNER(AC1) = NCPNER
     PNTDIS(AC1) = NCPDIS
  ENDDO
  IF (PRNLEV > 3) THEN
     WRITE(OUTU,'(A,I8)')'TOTAL NEAR CLUSTER PAIRS',NCPNER
     WRITE(OUTU,'(A,I8)')'TOTAL DISTANT CLUS PAIRS',NCPDIS
  ENDIF
  !
  ! -----------------------------------------------------
  ! The final loop, which does the final atom-atom distance
  ! calculations and exclusions, and generates the non-bonded
  ! list.
  !
  NNNB = 0
  !
  !  First loop over all clusters.  For each cluster
  !  loop over all the atoms it contains.
#if KEY_PARALLEL==1
  DO AC1 = MYNODP,NACTC,NUMNOD
#else /**/
     DO AC1 = 1,NACTC
#endif 
        IQ=ACLHI(AC1)
        IS=IQ - ACLPTN(AC1) + 1
        IF (ACLPTN(AC1) <= 0) CALL DIE
        DO BB = IS,IQ
           AT1 = JACLS(BB)
           MAXDIF = NUEXCA(AT1)
           ! store the atom1-dependent part of the
           ! exclusion (table) array pointers and the
           ! coordinates of atom1
           B1 = ATEXPT(AT1)
           FIXED = IMOVE(AT1) > 0
           X1 = X(AT1)
           Y1 = Y(AT1)
           Z1 = Z(AT1)
           !  Do inTRAcluster atom comparisons -------------------
           ! Added line below--RJP 7.20.99
           STRTAT=BB+1
           DO DD = STRTAT,IQ
              AT2 = JACLS(DD)
              IF((.NOT.FIXED).OR.(IMOVE(AT2) <= 0)) THEN
                 XX1 = X(AT2) - X1
                 YY1 = Y(AT2) - Y1
                 ZZ1 = Z(AT2) - Z1
                 IF (XX1*XX1 + YY1*YY1 + ZZ1*ZZ1  <=  CTNBSQ) THEN
                    !   check exclusions for this pair:
                    DIFF = AT2-AT1
                    IF (DIFF > MAXDIF) THEN
                       NNNB = NNNB + 1
                       JNB(NNNB) = AT2
                    ELSE
                       IF (EXCLAR(B1 + DIFF) == 1) THEN
                          NNNB = NNNB + 1
                          JNB(NNNB) = AT2
                       ELSE IF (EXCLAR(B1 + DIFF) == -1) THEN
                          NNNB = NNNB + 1
                          JNB(NNNB) =-AT2
                       ENDIF
                    ENDIF !if possibly an exclusion
                 ENDIF !if less than cutoff
              ENDIF
           ENDDO
           !  Do nearby inTERcluster comparisons (not requiring
           ! atom-atom distance calculations) ---------------------------------
           TT = PNTNER(AC1) - PARTNN(AC1) + 1
           DO CC = TT,PNTNER(AC1)
              AC2 = CPNER(CC)
              IQ2 =ACLHI(AC2)
              IS2=IQ2 - ACLPTN(AC2) + 1
              IF (ACLPTN(AC2) <= 0) CALL DIE
              DO DD = IS2,IQ2
                 AT2 = JACLS(DD)
                 IF((.NOT.FIXED).OR.(IMOVE(AT2) <= 0)) THEN
                    DIFF = AT2-AT1
                    IF (DIFF > MAXDIF) THEN
                       NNNB = NNNB + 1
                       JNB(NNNB) = AT2
                    ELSE
                       IF (EXCLAR(B1 + DIFF) == 1) THEN
                          NNNB = NNNB + 1
                          JNB(NNNB) = AT2
                       ELSE IF (EXCLAR(B1 + DIFF) == -1) THEN
                          NNNB = NNNB + 1
                          JNB(NNNB) =-AT2
                       ENDIF
                    ENDIF
                 ENDIF
              ENDDO
           ENDDO
           ! Do distant inTERcluster comparisons (requiring
           ! atom-atom distance calculations) --------------------------------------
           TT = PNTDIS(AC1) - PARTND(AC1) + 1
           DO CC = TT,PNTDIS(AC1)
              AC2 = CPDIS(CC)
              IQ2 =ACLHI(AC2)
              IS2=IQ2 - ACLPTN(AC2) + 1
              IF (ACLPTN(AC2) <= 0) CALL DIE
              DO DD = IS2,IQ2
                 AT2 = JACLS(DD)
                 IF((.NOT.FIXED).OR.(IMOVE(AT2) <= 0)) THEN
                    XX1 = X(AT2) - X1
                    YY1 = Y(AT2) - Y1
                    ZZ1 = Z(AT2) - Z1
                    IF (XX1*XX1 + YY1*YY1 + ZZ1*ZZ1  <=  CTNBSQ) THEN
                       !   check exclusions for this pair:  ----------------------
                       DIFF = AT2-AT1
                       IF (DIFF > MAXDIF) THEN
                          NNNB = NNNB + 1
                          JNB(NNNB) = AT2
                       ELSE
                          IF (EXCLAR(B1 + DIFF) == 1) THEN
                             NNNB = NNNB + 1
                             JNB(NNNB) = AT2
                          ELSE IF (EXCLAR(B1 + DIFF) == -1) THEN
                             NNNB = NNNB + 1
                             JNB(NNNB) =-AT2
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDDO
           ENDDO
           INBLO(AT1) = NNNB
           IF ((NNNB + NACTVE) > MAXJNB) THEN
              NNNB = MAXJNB + 1
#if KEY_PARALLEL==1 /*maxjnbsze*/
              IF(NUMNOD > 1) THEN
                 WRITE(OUTU,'(A)') &
                      ' NON BOND LIST ARRAY SIZE EXCEEDED; INCREASE NBSCALE'
                 CALL WRNDIE(-5,'<NBNDCCO>','ARRAY SIZE LIMIT EXCEEDED')
              ENDIF
#endif /* (maxjnbsze)*/
              RETURN
           ENDIF
        ENDDO
     ENDDO
     !
     ! since the non-bonded list has been filled only for active atoms
     ! (rest of INBLO(I) are zero) fill in the rest as appropriate --------
     !
     DO I = 2,NATOM
        IF (INBLO(I) == 0) INBLO(I) = INBLO(I-1)
     ENDDO
     !----------------------------------------------------------------------
     !---- Flag success and print statistics -------------------------------
     !
     CMPLTD=.TRUE.
     !
     IF (PRNLEV >= 2) &
          WRITE(OUTU,*)' NBNDCCF found: ', NNNB, ' atom pairs'
     !
     PRNLEV = PLVOLD
     !
#endif /* (yesbycc)*/
     RETURN
  END SUBROUTINE NBNDCCF
end module bycc_mod

      SUBROUTINE NBNDCCO(NNNB,JNB,MAXJNB,INBLO,X,Y,Z, &
       INB14,IBLO14,CUTNB,CPNER,CPDIS,NCPNER,NCPDIS, &
       CMPLTD,PNTNER,PNTDIS,PARTNN,PARTND,ORDCBN,XOO, &
       YOO,ZOO,DUMMYX,HAFMAR,MEANX,MEANY,MEANZ,LSTN0, &
       NUMMP,HIMPRTN,WRKAR,MMM,LSTM0,CNTN,XO,YO,ZO, &
#if KEY_TSM==1 /*tsmarg*/
       LTSM,REACLS,PRODLS, &
#endif /* (tsmarg)*/
       MXNCUB,EXCLAR,ATEXPT,NUEXCA,ATSX,ATSY,ATSZ)
#if KEY_NO_BYCC==0 /*yesbycc*/
!
!-----------------------------------------------------------------------
!  Same as NBNDCCF, but more compatibilities
!
!  Spatial decomposition modifications added July 2005
!                                          --RJP and Milan Hodoscek
!
  use chm_kinds
  use dimens_fcm
  use parallel
#if KEY_SPACDEC==1
  use spacdec        
#endif
  use memory
  use psf
  use stream
  use actclus_mod
  use contrl
  use comand
  use corman3,only:squareup
  use deriv  !temporary
  use machutil,only:die
#if KEY_MPI==1 && KEY_PARALLEL==1
  use mpi      
#endif
      implicit none
! *****************************************************************************
! PASSED VARIABLES
      INTEGER NNNB,MAXJNB,NCPNER,NCPDIS
      INTEGER JNB(*),INBLO(*)
      real(chm_real) X(*),Y(*),Z(*)
      INTEGER INB14(*),IBLO14(*)
      real(chm_real) CUTNB
      LOGICAL CMPLTD
      INTEGER MXNCUB ! maximum number of cubes in system
      INTEGER EXCLAR(*),ATEXPT(*),NUEXCA(*)  !exclusion table
      real(chm_real)  ATSX(*),ATSY(*),ATSZ(*) !for heuristic update
!
! *****************************************************************************
! "LOCAL" ARRAYS PASSED FOR MEMORY ALLOCATION ONLY
!
      INTEGER CPNER(*),CPDIS(*) !2nd-cluster lists (distant & near)
      INTEGER PNTNER(*),PNTDIS(*) ! pointers into CPNER and CPDIS
      INTEGER PARTND(*) !number of distant partners for a cluster
      INTEGER PARTNN(*) !number of nearby partners for a cluster
      INTEGER ORDCBN(*) ! ordinal number cube position of cluster
      INTEGER XOO(*),YOO(*),ZOO(*) !cubical grid coord of cluster
      INTEGER DUMMYX(*) !dummy storage array
      real(chm_real) HAFMAR(*) !largest center-to-atom distance in a cluster
      real(chm_real) MEANX(*),MEANY(*),MEANZ(*) !Cartes coord of cluster
      INTEGER LSTN0(*) !atom of next-highest number in the same cube
!  Cube arrays
      INTEGER NUMMP(*)    ! # of non-empty partner-cubes in surroundings
      INTEGER HIMPRTN(*)  ! pointer into partner-cube list (WRKAR)
      INTEGER WRKAR(*) ! partner-cube list
      INTEGER MMM(*)      ! ordinal number of each non-empty cube on grid
      INTEGER LSTM0(*)    ! highest atom number contained in a cube
      INTEGER CNTN(*)     ! # of clusters contained in a cube
      INTEGER XO(*),YO(*),ZO(*) !cubical grid coord of cubes
! ****************************************************************************
! LOCAL SCALARS
!  for exclusions
      INTEGER DIFF ! diffrnce between atom numbers in excluded pairs
      INTEGER MAXDIF !max diffrnce between atom numbers in excluded pairs
      INTEGER EXCL14 !exclusion flag (if==0, atom pair is excluded)
      INTEGER ELEN !table index
!  for cubes
      real(chm_real) CUBL,CUBNEED !cube lngth; necessary cube lngth for v. larg system
      INTEGER TOTCUB  !total number of cubes in system
      INTEGER FLDM !total number of non-empty cubes
      INTEGER NUMBX,NUMBY,NUMBZ ! dimensions of system in cube-lengths
      INTEGER NXY               ! dimension of system in XY plane
      real(chm_real) MAXX(3),MINX(3),MAXY(3),MINY(3),MAXZ(3),MINZ(3)
      INTEGER MNUM !size of cube-cube partner list (WRKAR)
      INTEGER PP,ZER ! temp storage of ordinal cube number
!  for clusters
      real(chm_real) DXT,DYT,DZT !sum of atm coordinates for avrgs in clusters
      real(chm_real) DXN,DYN,DZN !distnce components of atoms from center of clusters
      real(chm_real) XAVG,YAVG,ZAVG  ! ave position of all clusters
      real(chm_real) XSUM,YSUM,ZSUM ! summed coordinates for averages over all clusters
      real(chm_real) MARG !cluster margin (distance added to cutnb for testing)
      real(chm_real) OMARG
      real(chm_real) IMAR
      real(chm_real) IMARSQ !interclus dist below which interatomic dist MUST be <CUTNB
      real(chm_real) MARGSQ !interclus dist below which interatomic dist possibly<CUTNB
      real(chm_real) DMAX1,DMAX2 !1st and 2nd highest half-margin distances
!  for atoms
      real(chm_real) DSQ ! temp storage of squared distance
      real(chm_real) XX1,YY1,ZZ1,X1,Y1,Z1  !temp coordinate storage
      real(chm_real) CTNBSQ ! square of cutoff distance
      LOGICAL FIXED  !fixed atom (IMOVE  >  0)
!  various indices and other
      real(chm_real) GRMAX,STOR,XR,YR,ZR !temporary storage variables
      real(chm_real) MAC1X,MAC1Y,MAC1Z,XM,YM,ZM !temporary coordnate storage
      INTEGER B1,B2,B3,R,STRTAT
      INTEGER ACG,IS2,IQ2,ACT,AT1,AT2,ACC,AC1,AC2
      INTEGER ACC1,AC,BB,DD
      INTEGER IS,IQ,NAT,I,N,J,AT,T,KK,M,CC,TT,UU,SS
      INTEGER PLVOLD
      INTEGER ITEMP,NPR
! added in spat decomp, but needs to be in serial as well:
      INTEGER MAXCUB
      PARAMETER(MAXCUB=20000)
      INTEGER CCC,TMPCUB(MAXCUB),FLDCNT,CUBE,SFCUBE(MAXCUB+1)
      INTEGER ANINTG,DUMCUB,AC1N,AC2N,COUNT
      PARAMETER(ANINTG=9999999)
#if KEY_SPACDEC==1 /*spacdec0*/
!     for spatial decomposition
      INTEGER MXCLS
      PARAMETER(MXCLS=200000)
      real(chm_real) TEMSZE
      INTEGER BLCSZE,NODE,ISTART(MAXNODE),IEND(MAXNODE)
      INTEGER START,END
      INTEGER CPUNUM(MAXCUB),CPU1,CPU2,LASTCPU1
      INTEGER NLIST1,LIST1(MAXCUB),NLIST2,LIST2(MAXCUB), &
       NLIST3,LIST3(MAXCUB),NLIST4,LIST4(MAXCUB)
      INTEGER MAXCPUPAIR,KKK
      PARAMETER (MAXCPUPAIR=100000)
      INTEGER LOCCPUPR,NCPUPAIR,III,JJJ,CPUPRLST(MAXCPUPAIR)
      INTEGER CPUPNTHI(MAXNODE),CPUPRMANY(MAXNODE)
      INTEGER NCPU27(27),LCPU27(27),CPU27(27,MAXCUB)
      INTEGER WORK1(MAXCUB),WORK2(MAXCUB),NVAL1,NVAL2
      INTEGER ROUND,CLEAN
      INTEGER CPUPRFLG(MAXNODE*40),NODE1,NODE2,PNT
      INTEGER CP2FLG(MAXNODE), &
       NPAIR,ACCN,BBB,ATM,IQQ,ISS
      INTEGER LOCATM,BEGIN
      INTEGER CLS,LOCACFLG(MXCLS)
      LOGICAL DONE,HIT 
      real(chm_real) DM1ARR(MAXNODE),DM2ARR(MAXNODE)
      real(chm_real) MAXXAR(MAXNODE),MAXYAR(MAXNODE),MAXZAR(MAXNODE), &
       MINXAR(MAXNODE),MINYAR(MAXNODE),MINZAR(MAXNODE)
      INTEGER TRIAL,LASTDIR,MAXTRIAL
      real(chm_real) MAXPAIRS,MINPAIRS
      INTEGER CUBCNT,NMYACCL,MYACCL(MXCLS)
      LOGICAL RIGHT
      real(chm_real) GOAL,RUNNING,TEMP,ADDNAT,BALFAC,INCB
!      DATA PASSES/0/
      INTEGER TOTLBC,ESTLBC(MAXCUB),SUMCON
      INTEGER NCPUCUB(27),CUBCPU(27,MAXCUB),CPUCUB(27,MAXCUB)
      INTEGER LOWCPU,NLOCCUB1,NLOCCUB2,WORKC1(MAXCUB),WORKC2(MAXCUB), &
       LISTC1(MAXCUB)
      INTEGER LOWPNT,LOCATFLG(MAXA),NMYSATM,MM,ITH,IPHI,IIMP
      INTEGER ATLDCLS(MXCLS)
      real(chm_real) CRITER 
      SAVE LOCACFLG,ATLDCLS,LOCATFLG  !put in common 
!
#endif /* (spacdec0)*/
!
! options
#if KEY_TSM==1 /*tsmdecl*/
      LOGICAL LTSM
      INTEGER REACLS(*),PRODLS(*)
#endif /* (tsmdecl)*/
      LOGICAL DOIT
!
! *****************************************************************************
! ************* End of variables for BYCC *************************************
! *****************************************************************************
!
      PLVOLD = PRNLEV
      IF (DYNAMQ.OR.MINXYZ) PRNLEV = 3
!
      IF(PRNLEV > 3) WRITE(OUTU,'(A)') 'Using C-IN-C search'
!     Store atom positions
! and initialize the non-bonded list base array
! unassigned elements MUST equal zero later in algorithm.
#if KEY_SPACDEC==1
      MAXTRIAL = 100
      IF(QICPUMAP) THEN
         call chmalloc('nbndcc.src','NBNDCCO','ICPUMAP',NATOM,intg=ICPUMAP)
         QICPUMAP=.FALSE.
      ENDIF
#endif 
      DO I=1,NATOM
       INBLO(I) = 0  !base array
       ATSX(I)=X(I)
       ATSY(I)=Y(I)
       ATSZ(I)=Z(I)
#if KEY_SPACDEC==1
       IF((.NOT.LSPACFRST).AND.(LOCATFLG(I) /= 1)) THEN
        X(I) = 0
        Y(I) = 0
        Z(I) = 0
       ENDIF 
       LOCATFLG(I) = 0  
#if KEY_SPACDEC==1
       icpumap(i) = -1     
#endif
#endif 
      ENDDO
!      CALL PRCPUMAP(ICPUMAP,NATOM,PASSES,MYNODP)
      CTNBSQ = CUTNB*CUTNB
!  some setup for spatial decomposition
!-- Calculate the average positions of clusters and the -----------------
! -- cluster dimensions and cluster margin necessary   ------------------
! parallel:  the first time through, use all the cluster positions
! to calculate minx, maxx, dmax1,dmax2.
! for subsequent passes use only MY cluster positions.
! later on in the code, need to save the cluster information
! for "MYNOD" for use here in next update
! need to communicate hafmar(), minx(),maxx(),dmax1,dmax2
      DMAX1 = 0
      DMAX2 = 0
! --added error message below RJP
      IF (NACTC == 0) CALL WRNDIE(-5,'<NBNDCCO>', &
       'NO ACTIVE CLUSTERS DEFINED')
      DO I = 1,NACTC
#if KEY_SPACDEC==1 /*spacdec10*/
         IF(LSPACFRST) THEN !if first pass
            LOCACFLG(I)=1  
         ENDIF 
#endif /* (spacdec10)*/
! ------------------------------------
! initialize some cluster-based arrays
         CPNER(I) = 0
         CPDIS(I) = 0
         PNTNER(I) = 0
         PNTDIS(I) = 0
         PARTND(I) = 0
         PARTNN(I) = 0
         ORDCBN(I) = 0
         XOO(I) = 0
         YOO(I) = 0
         ZOO(I) = 0
         DUMMYX(I) = 0
         HAFMAR(I) = 0
         MEANX(I) = 0
         MEANY(I) = 0
         MEANZ(I) = 0
! -------------------------------------
! for spatial decomp, you just want to loop over
! MY active clusters, not all, unless first pass
! if first pass, include all clusters
#if KEY_SPACDEC==1
         IF(LOCACFLG(I) == 1) THEN  /*(locacflg)*/
#endif
         IQ= ACLHI(I)
         IS=IQ - ACLPTN(I) + 1
         NAT = ACLPTN(I)
         IF (NAT <= 0) CALL DIE
         DXT=0
         DYT=0
         DZT=0
         DO BB = IS,IQ
           J = JACLS(BB)
            DXT=DXT+X(J)
            DYT=DYT+Y(J)
            DZT=DZT+Z(J)
         END DO
         MEANX(I)=DXT/NAT
         MEANY(I)=DYT/NAT
         MEANZ(I)=DZT/NAT
         GRMAX = 0
         DO BB = IS,IQ
          J = JACLS(BB)
          DXN=X(J) - MEANX(I)
          DXN = DXN*DXN
          DYN=Y(J) - MEANY(I)
          DYN = DYN*DYN
          DZN=Z(J) - MEANZ(I)
          DZN = DZN*DZN
          DSQ = DXN+DYN+DZN
          STOR = SQRT(DSQ)
          IF (STOR > GRMAX) THEN
           GRMAX = STOR
          ENDIF
          IF(DSQ > DMAX1) THEN
            DMAX2 = DMAX1
            DMAX1 = DSQ
          ELSE IF (DSQ > DMAX2) THEN
            DMAX2 = DSQ
          ENDIF
         ENDDO  !over BB
         HAFMAR(I) = GRMAX   
#if KEY_SPACDEC==1
         ENDIF  /*(locacflg) */
#endif
      ENDDO
#if KEY_SPACDEC==1 /*spacdec30*/
! need to communicate DMAX1, DMAX2 and then take minima and
! maxima of whole set
      IF(.NOT.LSPACFRST) THEN
      DO NODE = 1,NUMNOD
        DM1ARR(NODE) = 0
        DM2ARR(NODE) = 0
! for minima
        MINXAR(NODE) = 0
        MINYAR(NODE) = 0
        MINZAR(NODE) = 0
        MAXXAR(NODE) = 0
        MAXYAR(NODE) = 0
        MAXZAR(NODE) = 0
      ENDDO      
      DM1ARR(MYNODP) = DMAX1
      DM2ARR(MYNODP) = DMAX2
      CALL GCOMB(DM1ARR,NUMNOD)
      CALL GCOMB(DM2ARR,NUMNOD)
      CALL GCOMB(HAFMAR,NACTC)
      CALL GCOMB(MEANX,NACTC)
      CALL GCOMB(MEANY,NACTC)
      CALL GCOMB(MEANZ,NACTC)
      CALL GCOMB(ATLDCLS,NACTC)
      CALL GCOMB(X,NATOM)
      CALL GCOMB(Y,NATOM)
      CALL GCOMB(Z,NATOM)
      DMAX1 = -ANINTG
      DMAX2 = -ANINTG
      DO III = 1,NUMNOD
       IF(DM1ARR(III) > DMAX1) THEN
        DMAX2 = DMAX1
        DMAX1 = DM1ARR(III)
       ELSE IF (DM1ARR(III) > DMAX2) THEN
        DMAX2 = DM1ARR(III)
       ENDIF
       IF(DM2ARR(III) > DMAX1) THEN
        DMAX2 = DMAX1
        DMAX1 = DM2ARR(III)
       ELSE IF (DM2ARR(III) > DMAX2) THEN
        DMAX2 = DM2ARR(III)
       ENDIF
      ENDDO 
      ENDIF !if not the first pass
#endif /* (spacdec30)*/
      MARG = SQRT(DMAX2) + SQRT(DMAX1)
      IF (PRNLEV > 3) THEN
        WRITE (OUTU,'(A,F6.4)') 'CALCed CLUS MARGIN = ', &
       MARG
      ENDIF
      IF (MARG > 5) THEN
       CALL WRNDIE(1,'<NBNDCCO>', &
       'LARGE CLUSTERS PRESENT, MARGIN > 5A')
      ENDIF
      IF (MARG > MARGL) THEN
       CALL WRNDIE(1,'<NBNDCCO>', &
       'CLUSTER MARGINS EXCEED SPECIFIED LIMIT')
       IF (PRNLEV >= 2) THEN
        WRITE(OUTU,*) '***************************'
        WRITE(OUTU,*) 'WARNING: CLUSTER MARGINS EXCEED '
        WRITE(OUTU,*) 'SPECIFIED LIMIT'
        WRITE(OUTU,*) 'CLUS MARGIN SET TO LIMIT = ', &
       MARGL
        WRITE(OUTU,*) 'SOME ATOM PAIRS MAY BE LOST'
        WRITE(OUTU,*) '***************************'
        WRITE(OUTU,*) ' '
       ENDIF
       MARG = MARGL
      ENDIF
2     FORMAT(A,F8.2,A,F8.2,A,F8.2)
3     FORMAT(A,I6,A,F8.2)
4     FORMAT(A,F5.3)
!
      CMPLTD=.FALSE.            ! Will remain .FALSE. until very end
!
! --set cube length equal to (cut-off dist + margin width ) --
      CUBL = CUTNB + MARG
      IF (PRNLEV > 3) WRITE (OUTU,'(A,F6.3)') &
       'CUBE LENGTH ',CUBL
! --initialize sums for center of geometry calculation
! initialize coordinate limits -------------------------------
      XSUM = 0
      YSUM = 0
      ZSUM = 0
       DO KK = 1,3
        MINX(KK) = ANINTG
        MAXX(KK) = -ANINTG
        MINY(KK) = ANINTG
        MAXY(KK) = -ANINTG
        MINZ(KK) = ANINTG
        MAXZ(KK) = -ANINTG
       ENDDO
! -------- Loop over clusters to find center of geometry
! and coordinate limits (in terms of clusters)
! -------------------------------------------------------
       DO ACC = 1,NACTC
#if KEY_SPACDEC==1 /*spacdec35*/
        IF(LOCACFLG(ACC) == 1) THEN  
#endif /* (spacdec35)*/
        XSUM = XSUM + MEANX(ACC)
        YSUM = YSUM + MEANY(ACC)
        ZSUM = ZSUM + MEANZ(ACC)
        IF (MEANX(ACC) > MAXX(1)) THEN
           MAXX(1) = MEANX(ACC)
           MAXX(2) = MEANY(ACC)
           MAXX(3) = MEANZ(ACC)
        ENDIF
        IF (MEANX(ACC) < MINX(1)) THEN
           MINX(1) = MEANX(ACC)
           MINX(2) = MEANY(ACC)
           MINX(3) = MEANZ(ACC)
        ENDIF
        IF (MEANY(ACC) > MAXY(2)) THEN
           MAXY(1) = MEANX(ACC)
           MAXY(2) = MEANY(ACC)
           MAXY(3) = MEANZ(ACC)
        ENDIF
        IF (MEANY(ACC) < MINY(2)) THEN
           MINY(1) = MEANX(ACC)
           MINY(2) = MEANY(ACC)
           MINY(3) = MEANZ(ACC)
        ENDIF
        IF (MEANZ(ACC) > MAXZ(3)) THEN
           MAXZ(1) = MEANX(ACC)
           MAXZ(2) = MEANY(ACC)
           MAXZ(3) = MEANZ(ACC)
        ENDIF
        IF (MEANZ(ACC) < MINZ(3)) THEN
           MINZ(1) = MEANX(ACC)
           MINZ(2) = MEANY(ACC)
           MINZ(3) = MEANZ(ACC)
        ENDIF
#if KEY_SPACDEC==1
      ENDIF   /* if locacflg is 1*/
#endif
      ENDDO
! ***************************************************************
! the xmin extremum is MINX(1),
! the xmax extremum is MAXX(1), the ymin extremum is MINY(2),
! the ymax extremum is MAXY(2), the zmin extremum is MINZ(3),
! the zmax extremum is MAXZ(3)
!  communicate MINX MAXX
#if KEY_SPACDEC==1 /*spacdec38*/
      IF(.NOT.LSPACFRST) THEN
       DO NODE = 1,NUMNOD
        MINXAR(NODE) = 0
        MINYAR(NODE) = 0
        MINZAR(NODE) = 0
        MAXXAR(NODE) = 0
        MAXYAR(NODE) = 0
        MAXZAR(NODE) = 0
       ENDDO
       MINXAR(MYNODP) = MINX(1)
       MINYAR(MYNODP) = MINY(2)
       MINZAR(MYNODP) = MINZ(3)
       MAXXAR(MYNODP) = MAXX(1)
       MAXYAR(MYNODP) = MAXY(2)
       MAXZAR(MYNODP) = MAXZ(3)
       CALL GCOMB(MINXAR,NUMNOD)
       CALL GCOMB(MINYAR,NUMNOD)
       CALL GCOMB(MINZAR,NUMNOD)
       CALL GCOMB(MAXXAR,NUMNOD)
       CALL GCOMB(MAXYAR,NUMNOD)
       CALL GCOMB(MAXZAR,NUMNOD)
       DO KK = 1,3
        MINX(KK) = ANINTG
        MAXX(KK) = -ANINTG
        MINY(KK) = ANINTG
        MAXY(KK) = -ANINTG
        MINZ(KK) = ANINTG
        MAXZ(KK) = -ANINTG
       ENDDO
       DO III = 1,NUMNOD
        IF(MINXAR(III) < MINX(1)) THEN
         MINX(1) = MINXAR(III)
        ENDIF
        IF(MINYAR(III) < MINY(2)) THEN
         MINY(2) = MINYAR(III)
        ENDIF
        IF(MINZAR(III) < MINZ(3)) THEN
         MINZ(3) = MINZAR(III)
        ENDIF
        IF(MAXXAR(III) > MAXX(1)) THEN
         MAXX(1) = MAXXAR(III)
        ENDIF
        IF(MAXYAR(III) > MAXY(2)) THEN
         MAXY(2) = MAXYAR(III)
        ENDIF
        IF(MAXZAR(III) > MAXZ(3)) THEN
         MAXZ(3) = MAXZAR(III)
        ENDIF 
       ENDDO
      ENDIF  !if not first pass
#endif /* (spacdec38)*/
! **************************************************************
      XAVG = XSUM/NACTC  !xavg, yavg, zavg not communicated
      YAVG = YSUM/NACTC
      ZAVG = ZSUM/NACTC
5     FORMAT(A,F8.2,A,F8.2)
      DO KK = 1,3
       IF ((MAXX(KK) >= ANINTG).OR.(MAXY(KK).GE.ANINTG).OR. &
        (MAXZ(KK) >= ANINTG)) THEN
        CALL WRNDIE(-5,'<NBNDCCO>', &
      'NBNDCCO> SOME COORDNTS >= LIMIT (UNDEFINED)')
       ENDIF
      ENDDO
      NUMBX = INT((MAXX(1)-MINX(1))/CUBL) + 1
      NUMBY = INT((MAXY(2)-MINY(2))/CUBL) + 1
      NUMBZ = INT((MAXZ(3)-MINZ(3))/CUBL) + 1
      TOTCUB = NUMBX*NUMBY*NUMBZ
! ----------------------------------------------------------------
! ****************************************************************
! ----------------------------------------------------------------
#if KEY_SPACDEC==1
      IF (NUMNOD == 1) THEN  
#endif
! To avoid communication, flag the next part to be skipped
! if numnod is gt 1 in spacdec only.
!
! If cubical grid is too large, try to reorient molecule -------
      IF (TOTCUB > MXNCUB) THEN
       IF (PRNLEV > 3) THEN
         WRITE(OUTU,'(A,I8)') 'TOTAL CUBES = ', TOTCUB, &
       'EXCEED LIM = ', MXNCUB
         WRITE(OUTU,'(A)') 'TRYING TO SQUARE UP STRUCTURE'
       ENDIF
14     FORMAT(A,I10)
15     FORMAT(A,F8.2)
       DO I = 1,NACTC
         DUMMYX(I) = I
       ENDDO
!
       CALL SQUAREUP(MAXX,MINX,MAXY,MINY,MAXZ,MINZ,XAVG, &
                     YAVG,ZAVG,MEANX,MEANY,MEANZ,NACTC,DUMMYX,NACTC)
!
#if KEY_SPACDEC==1
       ENDIF 
#endif
! ----------------------------------------------------------------
! ****************************************************************
! ----------------------------------------------------------------
! end of skipped part for spatial decomp
       XR = MAXX(1)-MINX(1)
       YR = MAXY(2)-MINY(2)
       ZR = MAXZ(3)-MINZ(3)
       NUMBX = INT(XR/CUBL) + 1
       NUMBY = INT(YR/CUBL) + 1
       NUMBZ = INT(ZR/CUBL) + 1
       TOTCUB = NUMBX*NUMBY*NUMBZ
       IF (PRNLEV > 3) THEN
         WRITE(OUTU,14) 'TOTAL CUBES (AFTER SQUARE) = ', &
         TOTCUB
       ENDIF
!
! -- IF grid is still too large, adjust cube length ----------------
       IF (TOTCUB > MXNCUB) THEN
        CUBNEED = CUBL
        DO WHILE(TOTCUB > MXNCUB)
         CUBNEED = ((XR + CUBNEED)*(YR + CUBNEED)* &
        (ZR + CUBNEED))
         CUBNEED = ((CUBNEED/MXNCUB)**0.33333333)
         NUMBX = INT(XR/CUBNEED) + 1
         NUMBY = INT(YR/CUBNEED) + 1
         NUMBZ = INT(ZR/CUBNEED) + 1
         TOTCUB = NUMBX*NUMBY*NUMBZ
         WRITE(OUTU,'(A,F6.3,A,I9)') 'CNEED ',CUBNEED, &
        ' TOTCUB ',TOTCUB
        ENDDO
        CUBL = CUBNEED
        IF (PRNLEV >= 2) THEN
         WRITE(OUTU,15) 'EXPANDING CUBL to CUBNEED= ', &
        CUBL
        ENDIF
       ENDIF ! overall grid still large after reorient
      ENDIF ! overall grid too large
!
      IF (PRNLEV > 3) THEN
        WRITE (OUTU,'(A)') &
            'NBNDCCO Building particle interaction list using grid'
        WRITE (OUTU,'(A,I8)')' Number of active atoms         =', NACTVE
        WRITE (OUTU,'(A,I8)')' Number of active clusters      =', NACTC
        WRITE (OUTU,'(A,I8)')' Number of cells in X dimension =', NUMBX
        WRITE (OUTU,'(A,I8)')' Number of cells in Y dimension =', NUMBY
        WRITE (OUTU,'(A,I8)')' Number of cells in Z dimension =', NUMBZ
        WRITE (OUTU,'(A,I8)')' Number of cells, total         =', TOTCUB
      ENDIF
      IF (PRNLEV > 3) WRITE (OUTU,'(A,F7.2)') &
       ' Cell size                      =', CUBL
!  
      NXY = NUMBX*NUMBY
! ---initialize some cube arrays ------------------------------
       DO I = 1, TOTCUB
        LSTM0(I) = 0
        CNTN(I) = 0
        NUMMP(I) = 0
        HIMPRTN(I) = 0
        MMM(I) = 0
        XO(I) = 0
        YO(I) = 0
        ZO(I) = 0
        TMPCUB(I) = 0
       ENDDO
! -------------------------------------------------------------
! check to see which cubes are "filled" (non-empty) and
! for the filled ones, store:
! their positions on the cubical grid (XO,YO,ZO)
! and the number of clusters they contain (CNTN)
! Also, store the total number of filled cubes (FLDM)
       FLDM = 0
       DO AC = 1,NACTC
#if KEY_SPACDEC==1
        LOCACFLG(AC) = 0   /*init all local cluster flags for spacdec */
#endif
        LSTN0(AC) = 0
        PARTNN(AC) = 0
        PARTND(AC) = 0
        XOO(AC) = INT((MEANX(AC)-MINX(1))/CUBL)
        YOO(AC) = INT((MEANY(AC)-MINY(2))/CUBL)
        ZOO(AC) = INT((MEANZ(AC)-MINZ(3))/CUBL)
        ORDCBN(AC) = XOO(AC) + (YOO(AC))*NUMBX + &
       (ZOO(AC))*NXY + 1
        IF (ORDCBN(AC) > MXNCUB) THEN
         CALL WRNDIE(-5,'<NBNDCCO>', &
      'NBNDCCO> TOO MANY CUBES. INCRSE CUBL OR REDUCE SYSTEM')
        ENDIF
        M = ORDCBN(AC)
        IF (CNTN(M) == 0) THEN
         FLDM = FLDM + 1
         MMM(FLDM) = M
!  Note:  MMM is NOT sorted
         ZO(M) = ZOO(AC)
         YO(M) = YOO(AC)
         XO(M) = XOO(AC)
        ENDIF
        CNTN(M) = CNTN(M) + 1
       ENDDO
!
#if KEY_SPACDEC==1 /*spacdec40*/
!      WRITE(6,*) 'FLDM is ',FLDM,' NUMNOD is ',NUMNOD
      IF(FLDM < NUMNOD) THEN
       WRITE(6,*)
       WRITE(6,*) &
       '# OF PROCS EXCEEDS THE # OF NON-EMPTY CUBES IN SYSTEM'
      CALL WRNDIE(-5,'<NBNDCCO>', &
      'NBNDCCO> TOO FEW CUBES FOR NUMBER OF PROCESSES')
      ENDIF
#endif /* (spacdec40)*/
! --------------------------------------------------------------
! From Tom Ngo's "intertwined lists" algorithm.
! For each cube, the highest-numbered cluster contained in the
! cube is stored (in LSTM0). This cluster is also linked to the
! next highest cluster in number contained in the cube, such that
! LSTN0(high) = next highest. This is done for all clusters in
! a given cube. Thus the clusters within a particular cube are
! linked to each other in a "chain" and they are all linked
! to the cube number via the high cluster number.
       DO ACC1 = 1, NACTC
         M = ORDCBN(ACC1)
         LSTN0(ACC1) = LSTM0(M)
         LSTM0(M) = ACC1
       ENDDO
! -------------------------------------------------------------
!
! for spacdec
! here we need to determine how many cubes per cpu
! First we need to "sort" the mmm array, but not with an actual sort,
! which is slow.
! We flag the cubes that are filled and then create a compressed array. 
! this next section should be unprotected (for use in serial also)
!       
       DO CCC = 1,FLDM
         M = MMM(CCC)   !list of filled cubes
         TMPCUB(M) = 1  !flag the cubes that are filled
       ENDDO
       FLDCNT = 0
#if KEY_SPACDEC==1
       IF(.NOT.LSPACFRST) TOTLBC = 0  
#endif
       DO CUBE = 1,TOTCUB
         IF(TMPCUB(CUBE) == 1) THEN
          FLDCNT = FLDCNT + 1
          SFCUBE(FLDCNT) = CUBE  !sorted list of filled cubes
#if KEY_SPACDEC==1
          IF(.NOT.LSPACFRST) THEN
           ESTLBC(CUBE) = 0
! do atom statistics on cubes
           ACCN = 1
           ACC = LSTM0(CUBE)
           DO WHILE(ACCN <= CNTN(CUBE))
            ESTLBC(CUBE) = ESTLBC(CUBE) + ATLDCLS(ACC)
            TOTLBC = TOTLBC + ATLDCLS(ACC)
            ACC = LSTN0(ACC)
            ACCN = ACCN + 1
           ENDDO ! WHILE(ACCN <= CNTN(CUBE))
          ENDIF !if not first pass
#endif 
         ENDIF  !if tmpcub == 1
       ENDDO !loop over cubes 
#if KEY_SPACDEC==1
! Estimate contributions from each cube to load balance
! In initial guess, this is done by taking the product of the number 
! of atoms in the central cube and the number of atoms in all of its
! surrounding cubes.
! this is for the initial guess:
       IF(LSPACFRST) THEN
       TOTLBC = 0
       DO CC = 1,FLDM
        ZER = SFCUBE(CC)
        SUMCON = 0
        KK = 1
        DO WHILE(KK <= 27)
         IF(((BBX(KK)+XO(ZER)) >= 0).AND.((BBY(KK)+YO(ZER)).GE.0).AND. &
          ((BBZ(KK)+ZO(ZER)) >= 0)) THEN
!  Next line means the cube can't be out of bounds
          IF(((BBX(KK)+XO(ZER)) < NUMBX).AND. &
        ((BBY(KK)+YO(ZER)) < NUMBY) &
           .AND.((BBZ(KK)+ZO(ZER)) < NUMBZ)) THEN
           PP = BBX(KK) + BBY(KK)*NUMBX + BBZ(KK)*NXY + ZER
           SUMCON = SUMCON + CNTN(PP)
          ENDIF
         ENDIF
        KK = KK + 1
        ENDDO
         ESTLBC(ZER) = SUMCON*CNTN(ZER)
        TOTLBC = TOTLBC+ESTLBC(ZER)
       ENDDO
       ENDIF !if lspacfrst
#endif 
! SPATIAL ASSIGNMENT OF CPUS
! find out how to break up the list of filled cubes and assign to cpus
#if KEY_SPACDEC==1 /*spacdec50*/
       GOAL = TOTLBC/NUMNOD
       INCB = 0.2 !initial increment
       LASTDIR = 0  !0 is down, 1 is up
       BALFAC = 1
       RIGHT = .FALSE.
       TRIAL = 1
!       IF(MYNODP == 1) THEN
!       WRITE(6,*) 'GOAL ',GOAL,' TOTLBC ',TOTLBC,' NUMNOD ',
!     & NUMNOD
!       ENDIF
       DO WHILE (.NOT.RIGHT)
        NODE = 1
        ISTART(1) = 1
        IEND(1) = 0
        RUNNING = 0
        MAXPAIRS = -1.0E20
        MINPAIRS = 1.0E20
        CUBCNT = 0
        DO III = 1,FLDCNT
         ZER = SFCUBE(III)
         ADDNAT = ESTLBC(ZER)
         TEMP = BALFAC*(ADDNAT/2) + RUNNING
! the conditions for assigning a set of cubes to a cpu are
! 1) the number of atoms in the cpu meets the specified threshold
! 2) the cpu number is not higher than the total number
! 3) the number of cubes in the cpu is at least 1
         IF((TEMP > GOAL).AND.(NODE < NUMNOD).AND. &
       (CUBCNT > 0)) THEN
!          IF(PRNLEV > 2) THEN
!           WRITE(6,*) 'NODE ',NODE,' HAS ',RUNNING,
!     &  ' ATOMS'
!          ENDIF
!          RUNARR(NODE) = RUNNING !temporary ?
          NODE = NODE + 1
          IF(RUNNING > MAXPAIRS) MAXPAIRS = RUNNING
          IF(RUNNING < MINPAIRS) MINPAIRS = RUNNING
          RUNNING = 0
          CUBCNT = 0
          ISTART(NODE) = IEND(NODE-1)+1
          IEND(NODE) = IEND(NODE-1)
         ENDIF
         CUBCNT = CUBCNT + 1
         IEND(NODE) = IEND(NODE) + 1
         RUNNING = RUNNING + ADDNAT
        ENDDO  !loop over filled cubes
!        IF(MYNODP == 1) THEN
!        WRITE(6,*) 'HERE NODE IS ',NODE
!         WRITE(6,*) 'TRIAL ',TRIAL,' INCB ',INCB,' BALFAC ',
!     & BALFAC
!         WRITE(6,*) 'NUMNOD ',NUMNOD,' LAST CPU HAS ',RUNNING,
!     & ' ESTIMATED PAIRS ',
!     & 'WHILE MAX IS ',MAXPAIRS
!        WRITE(6,*) 'NODE ',NODE,' HAS ',RUNNING,
!     & ' ESTIMATED PAIRS'
!        ENDIF !mynodp
        CRITER = (RUNNING-GOAL)/GOAL
        IF((RUNNING > MAXPAIRS).AND.(NODE /= 1).AND. &
       (CRITER > 0.05).AND.(INCB.GT.0.00001)) THEN !last cpu too big
!         WRITE(6,*) 'RUNNING IS ',RUNNING,' NODE IS ',NODE
         IF(LASTDIR == 0) INCB = INCB/2
         BALFAC = BALFAC-INCB
         LASTDIR = 1
        ELSE IF ((NODE /= NUMNOD).AND.(TRIAL <= MAXTRIAL)) &
        THEN !not at end of cpu list
         IF(LASTDIR == 1) THEN
           INCB = INCB/2
         ELSE
           INCB = INCB*2
         ENDIF
         BALFAC = BALFAC+INCB
         LASTDIR = 0
        ELSE
         RIGHT = .TRUE.
        ENDIF
        TRIAL = TRIAL + 1
       ENDDO !while not right
!       RUNARR(NUMNOD) = RUNNING
       IF(RUNNING > MAXPAIRS) MAXPAIRS = RUNNING
       IF(RUNNING < MINPAIRS) MINPAIRS = RUNNING
!       IF(MYNODP == 1) THEN
!       WRITE(6,*) 'MAXPAIRS = ',MAXPAIRS,' MINPAIRS = ',MINPAIRS,
!     & ' AVERAGE = ',GOAL
!       WRITE(6,*) 'MAX/AVG ',MAXPAIRS/GOAL,' MIN/AVG ',MINPAIRS/GOAL
!       WRITE(6,*) 'NODE ',NODE,' HAS ',RUNNING,
!     & ' ATOMS'
!       DO NODE = 1,NUMNOD
!         WRITE(6,*) 'PASS ',PASSES,' NODE ',NODE,
!     & ' ISTART ',ISTART(NODE),
!     & ' IEND ',IEND(NODE)
!       ENDDO
!        DO NODE = 1,NUMNOD
!          IF(PRNLEV > 2) THEN
!           WRITE(6,*) 'NODE ',NODE,' HAS ',RUNARR(NODE),
!     &  ' ESTIMATED PAIRS'
!          ENDIF
!        ENDDO
!       ENDIF !mynodp
!       DO NODE = 1,NUMNOD
!         WRITE(6,*) 'PASS ',PASSES,' NODE ',NODE,
!     & ' ISTART ',ISTART(NODE),
!     & ' IEND ',IEND(NODE)
!       ENDDO
!
! note that ISTART and IEND give the INDICES of the SFCUBE
! array that point to the ordinal cube numbers, not the
! ordinal numbers themselves, which are what is really needed.
! Note also that for all spatial decomp section, we have 
! added 1 to all the cpu numbers
!
! -----------------------------------------------------------
! FOR LOCAL COMMUNICATION:
! Now loop through all the cubes by their ordinal number
! and and check the cubes in the local region
! first invert the list of cube<->cpu numbers
       NMYATM = 0
       DO NODE = 1,NUMNOD
        DO DUMCUB = ISTART(NODE),IEND(NODE) !loop over indices
         ZER = SFCUBE(DUMCUB) !sorted list of filled cube ordinal numbers
         CPUNUM(ZER) = NODE !list of cpu numbers indexed by ord cube number
!
         IF(NODE == MYNODP) THEN
          ACCN = 1
          ACC = LSTM0(ZER)
          DO WHILE(ACCN <= CNTN(ZER))
           LOCACFLG(ACC) = 1  !flag the clusters in this cpu
           IQQ=ACLHI(ACC)
           ISS=IQQ - ACLPTN(ACC) + 1
           IF (ACLPTN(ACC) <= 0) CALL DIE
           DO BBB = ISS,IQQ
            ATM = JACLS(BBB)
            NMYATM = NMYATM + 1
            icpumap(atm) = mynod
            MYATARR(NMYATM) = ATM
            LOCATFLG(ATM) = 1
           ENDDO
           ACC = LSTN0(ACC)
           ACCN = ACCN + 1
          ENDDO
         ENDIF
        ENDDO
       ENDDO 
!
       NMYSATM = 0
       DO JJJ = 1,NATOM
        IF(LOCATFLG(JJJ) == 1) THEN
          NMYSATM = NMYSATM + 1
          MYSATARR(NMYSATM) = JJJ 
        ENDIF
       ENDDO
! set up mappings for bonded energies
! for now this is in testing phase, the above works though...
!C       NMYBOND = 0
!C       NMYANGL = 0
!C       NMYDIHE = 0
!C       NMYIMPR = 0
!C       DO MM = 1,NBOND
!C         ATM = IB(MM)
!C         IF(LOCATFLG(ATM) == 1) THEN
!C           NMYBOND = NMYBOND + 1
!C           MYBOND(NMYBOND) = MM
!C         ENDIF 
!C       ENDDO
!C       DO ITH = 1,NTHETA
!C         ATM = IT(ITH)
!C         IF(LOCATFLG(ATM) == 1) THEN
!C           NMYANGL = NMYANGL + 1
!C           MYANGL(NMYANGL) = ITH
!C         ENDIF
!C       ENDDO
!C       DO IPHI = 1,NPHI
!C         ATM = IP(IPHI)
!C         IF(LOCATFLG(ATM) == 1) THEN
!C           NMYDIHE = NMYDIHE + 1
!C           MYDIHE(NMYDIHE) = IPHI
!C         ENDIF
!C       ENDDO
!C       DO IIMP = 1,NIMPHI
!C         ATM = IM(IIMP)
!C         IF(LOCATFLG(ATM) == 1) THEN
!C           NMYIMPR = NMYIMPR + 1
!C           MYIMPR(NMYIMPR) = IIMP
!C         ENDIF
!C       ENDDO
!
!       IF(MYNODP == 1) THEN
!       WRITE(6,*) 'NMYSATM ',NMYSATM,' NMYATM ',NMYATM
!       ENDIF
!        
!        DO III = 1,NMYATM
!        WRITE(6,*) 'MYNODP ',MYNODP,' INDEX ',III,' ATOM ',MYATARR(III)
!        ENDDO
! initialize some cpu-pair arrays and scalars 
       LASTCPU1 = 0
       NCPUPAIR = 0

       DO KKK = 1,27
        NCPU27(KKK) = 0
        LCPU27(KKK) = 0
        DO III = 1,MAXNODE
         CPU27(KKK,III) = 0
        ENDDO
! for cubes
       NCPUCUB(KKK) = 0
       ENDDO
#endif /* (spacdec50)*/
!
!--------------------------------------------------------------
!--------------------------------------------------------------
!--------------------------------------------------------------
! Loop over all filled cubes in the system
! For each filled cube, determine the set of filled cubes in its
! immediate surroundings and store the ordinal-number positions
! of these filled cubes in a linked list.
! For spatial decomposition, create a pairlist of communicating
! cpus.  Also, for my cpu, store the cube numbers that need
! to be exported to each of its partners in the cpu pairlist.
      MNUM = 0
#if KEY_SPACDEC==1 /*spacdec55*/
      SFCUBE(FLDM+1) = -1
      DO CC = 1,FLDM+1 !loop over filled cubes plus 1
#else /* (spacdec55)*/
      DO CC = 1,FLDM
#endif /* (spacdec55)*/
        ZER = SFCUBE(CC)        
!-----------------------------------------------------
!-----------------------------------------------------
#if KEY_SPACDEC==1 /*spacdec60*/
! for spatial decomp
        IF(CC == FLDM+1) THEN
          CPU1 = -1
        ELSE 
          CPU1 = CPUNUM(ZER)
        ENDIF
        IF((CPU1 /= LASTCPU1).AND.(LASTCPU1.NE.0)) THEN
! if changing cpu, collapse previous cpu's lists
! loop over all the frames, and combine the lists
! for each pair of successive frames
          NVAL1 = NCPU27(1)  
          DO III = 1,NVAL1
            WORK1(III) = CPU27(1,III)
          ENDDO
          DO KKK = 2,27 
           NVAL2 = NCPU27(KKK)
           DO III = 1,NVAL2
            WORK2(III) = CPU27(KKK,III)
           ENDDO
           CALL COMBINELIST(NVAL1,WORK1,NVAL2,WORK2, &
       NLIST1,LIST1)  
           NVAL1 = NLIST1
           DO III = 1,NVAL1
            WORK1(III) = LIST1(III)
           ENDDO 
          ENDDO
!
          LOCCPUPR = 0
!          WRITE(6,*) 'PASS ',PASSES,' MYNODP ',MYNODP,
!     & ' THE COMBINED LIST IS'
          DO III = 1,NLIST1
!            WRITE(6,*) 'III ',LIST1(III)
            LOCCPUPR = LOCCPUPR + 1
            NCPUPAIR = NCPUPAIR + 1
            CPUPRLST(NCPUPAIR) = LIST1(III)
          ENDDO   
! update the cpu pairlist pointers
          CPUPNTHI(LASTCPU1) = NCPUPAIR
          CPUPRMANY(LASTCPU1) = LOCCPUPR
! reinitialize the local counters
          DO III = 1,27
            NCPU27(III) = 0
            LCPU27(III) = 0
          ENDDO
        ENDIF  !if CPU1  /=  lastcpu1
        LASTCPU1 = CPU1
        IF(CC /= FLDM+1) THEN  !if not the last pass
#endif /* (spacdec60) spatial decomp*/
! -----------------------------------------------------------
! Loop over the 27 surrounding cubes in the immediate surroundings
! and store the ordinal-number positions of these filled cubes in
! a linked list
        KK = 1
        DO WHILE(KK <= 27)
         IF(((BBX(KK)+XO(ZER)) >= 0).AND.((BBY(KK)+YO(ZER)).GE.0).AND. &
          ((BBZ(KK)+ZO(ZER)) >= 0)) THEN
!  Next line means the cube can't be out of bounds
          IF(((BBX(KK)+XO(ZER)) < NUMBX).AND. &
        ((BBY(KK)+YO(ZER)) < NUMBY) &
           .AND.((BBZ(KK)+ZO(ZER)) < NUMBZ)) THEN
           PP = BBX(KK) + BBY(KK)*NUMBX + BBZ(KK)*NXY + ZER
           IF (CNTN(PP) > 0) THEN
            MNUM = MNUM + 1
            NUMMP(ZER) = NUMMP(ZER) + 1
            WRKAR(MNUM) = PP
!--------------------
#if KEY_SPACDEC==1 /*spacdec70*/
! for spacial decomp
! If the cpu number of the cube in the surroundings is different
! from this cpu, store it as a function of its position (NCPU27(KK))
! in the frame (KK).
            CPU2 = CPUNUM(PP)
            IF(CPU2 /= CPU1) THEN  !this means communication must occur
              IF(CPU2 /= LCPU27(KK)) THEN
                NCPU27(KK) = NCPU27(KK) + 1
                CPU27(KK,NCPU27(KK)) = CPU2
                LCPU27(KK) = CPU2
              ENDIF !cpu2 /= lcpu27
! if cpu1 is this cpu, record the central cube and the cpu to which
! it must be communicated (cpu2).
              IF(CPU1 == MYNODP) THEN
                NCPUCUB(KK) = NCPUCUB(KK) + 1
                CUBCPU(KK,NCPUCUB(KK)) = CPU2
                CPUCUB(KK,NCPUCUB(KK)) = ZER
!      WRITE(6,*) 'MNODP ',MYNODP,' LOOK: KK ',KK,' N ',NCPUCUB(KK),
!     & ' CPU ',
!     & CUBCPU(KK,III),
!     & ' CUB ',CPUCUB(KK,III)
! for testing purposes, erase stuff in adjacent cpus
!        GOTO 8654
!           ACCN = 1
!           ACC = LSTM0(PP)
!           DO WHILE(ACCN <= CNTN(PP))
!            IQQ=ACLHI(ACC)
!            ISS=IQQ - ACLPTN(ACC) + 1
!            IF (ACLPTN(ACC) <= 0) CALL DIE
!            DO BBB = ISS,IQQ
!             ATM = JACLS(BBB)
!             X(ATM) = 1000
!             Y(ATM) = 1000
!             Z(ATM) = 1000
!             COMMFLG(ATM) = 1 !temporary
!            ENDDO
!            ACC = LSTN0(ACC)
!            ACCN = ACCN + 1
!           ENDDO 
! 8654    CONTINUE
! end of testing
              ENDIF !cpu1 eq mynodp
            ENDIF !cpu2 eq cpu1
! end part for spatial decomp
!            CPU2 = CPUNUM(PP)
!            IF((CPU2 /= CPU1).AND.(CPU2.NE.LCPU27(KK))) THEN
!                NCPU27(KK) = NCPU27(KK) + 1
!                CPU27(KK,NCPU27(KK)) = CPU2
!                LCPU27(KK) = CPU2
!            ENDIF !cpu2 ne cpu1
! end part for spatial decomp
#endif /* (spacdec70)*/
! -------------------------------------------
           ENDIF !if cntn(pp)  >  0
          ENDIF !if test cube within limits
         ENDIF !if test cube within limits
         KK = KK + 1
        ENDDO !loop over surrounding cubes
        HIMPRTN(ZER) = MNUM
#if KEY_SPACDEC==1
        ENDIF !if not the last pass  
#endif
      ENDDO !loop over each filled cube in system
!--------------------------------------------------------------
!--------------------------------------------------------------
#if KEY_SPACDEC==1 /*spacdec78*/
! now process the cubes in mynodp
        DO NODE = 1,NUMNOD
          PRTATMHI(NODE) = 0
          PRTATMMNY(NODE) = 0
        ENDDO
        NPARTATM = 0
        LOWCPU = CPUPNTHI(MYNODP) - CPUPRMANY(MYNODP) + 1
!        WRITE(6,*) 'MYNODP ',MYNODP,' LOWCPU ',LOWCPU,' HICPU ',
!     & CPUPNTHI(MYNODP)
        DO PNT = LOWCPU,CPUPNTHI(MYNODP)
          NODE2 = CPUPRLST(PNT)
!         WRITE(6,*) 'MYNODP ',MYNODP,' PNT ',PNT,' NODE2 ',
!     & NODE2
          NLOCCUB1 = 0
          DO III = 1,NCPUCUB(1)
            CPU2 = CUBCPU(1,III)
            IF(CPU2 == NODE2) THEN
!      WRITE(6,*) 'MNDP ',MYNODP,' PNT ',PNT,' NODE2 ',NODE2,
!     & ' CUB ',CPUCUB(1,CUBE),' CPU2 ',CPU2,' KK ',1
               NLOCCUB1 = NLOCCUB1 + 1
               WORKC1(NLOCCUB1) = CPUCUB(1,III)
            ENDIF
          ENDDO
!         DO III = 1,NLOCCUB1
!         WRITE(6,*) 'NODE2 ',NODE2,' WORKC1 ',WORKC1(III)
!         ENDDO
          DO KKK = 2,27
           NLOCCUB2 = 0
           DO III = 1,NCPUCUB(KKK)
            CPU2 = CUBCPU(KKK,III)
            IF(CPU2 == NODE2) THEN
!        WRITE(6,*) 'MNDP ',MYNODP,' PNT ',PNT,' NODE2 ',NODE2,
!     & ' CUB ',CPUCUB(KKK,III),' CPU2 ',CPU2,' KK ',KKK
               NLOCCUB2 = NLOCCUB2 + 1
               WORKC2(NLOCCUB2) = CPUCUB(KKK,III)
            ENDIF
           ENDDO
           CALL COMBINELIST(NLOCCUB1,WORKC1,NLOCCUB2,WORKC2, &
       NLIST1,LISTC1)
           NLOCCUB1 = NLIST1
           DO III = 1,NLOCCUB1
            WORKC1(III) = LISTC1(III)
           ENDDO
          ENDDO !loop over 27 frames
!        WRITE(6,*) 'MYNDP ',MYNODP,' FOR PARTNER CPU ',NODE2,
!     & ' THE LIST IS:'
! make lists of atoms needing to be communicated to each partner cpu
!          LOCCLS = 0
          LOCATM = 0
          DO III = 1,NLIST1
           ZER = LISTC1(III)
!           WRITE(6,*) 'LOOK: MYNODP ',MYNODP,' NODE2 ',NODE2,
!     & ' III ',III,' CUB ',ZER,' CNTN ',CNTN(ZER)
           ACCN = 1
           ACC = LSTM0(ZER)
           DO WHILE(ACCN <= CNTN(ZER))
            IQQ=ACLHI(ACC)
            ISS=IQQ - ACLPTN(ACC) + 1
            IF (ACLPTN(ACC) <= 0) CALL DIE
            DO BBB = ISS,IQQ
             NPARTATM = NPARTATM + 1
             LOCATM = LOCATM + 1
             ATM = JACLS(BBB)
!             WRITE(6,*) 'IN LOOP: MYNODP ',MYNODP,' NODE ',NODE,
!     & ' ATOM ', ATM
             PRTATMLST(NPARTATM) = ATM
            ENDDO
            ACC = LSTN0(ACC)
            ACCN = ACCN + 1
           ENDDO !loop over clusters in partner cubes
          ENDDO !loop over cubes in partner cpu
          PRTATMHI(NODE2) = NPARTATM
          PRTATMMNY(NODE2) = LOCATM
        ENDDO  !loop over partner cpus
!        GOTO 1098
!        DO NODE = 1,NUMNOD
!          LOWPNT = PRTATMHI(NODE) - PRTATMMNY(NODE) + 1 
!          WRITE(6,*) 'NODE ',NODE,' LOWPNT ',LOWPNT,
!     & ' HIPNT ',PRTATMHI(NODE)     
!          DO PNT = LOWPNT,PRTATMHI(NODE)
!           WRITE(6,*) 'MYNODP ',MYNODP,' NODE ',NODE,
!     & ' ATOM ',PRTATMLST(PNT)
!          ENDDO
!        ENDDO
! 1098   CONTINUE
#endif /* (spacdec78)*/
! figure out the communication order
#if KEY_SPACDEC==1 /*spacdec80*/
!       WRITE(6,*) 'PASS ',PASSES,
!     & ' TOTAL NUMBER OF CPU PAIRS IS ',NCPUPAIR
       DO PNT = 1,NCPUPAIR
         CPUPRFLG(PNT) = 0
       ENDDO
       DO KKK = 1,27
         PARTNER(KKK) = 0  !0 here, since -1 passed in SPACSR
       ENDDO
       NROUND = 0
       NPAIR = 0
       DONE = .FALSE.
       DO WHILE(.NOT.DONE) 
        NROUND = NROUND + 1
        DO NODE = 1,NUMNOD
         CP2FLG(NODE) = 0
        ENDDO
        CLEAN = 0
        DO NODE1 = 1,NUMNOD  !loop over all cpus
         HIT = .FALSE.
          PNT = CPUPNTHI(NODE1) - CPUPRMANY(NODE1) + 1
          DO WHILE((.NOT.HIT).AND.(PNT <= CPUPNTHI(NODE1)))
           NODE2 = CPUPRLST(PNT)
           IF ((CP2FLG(NODE2) == 0).AND. &
       (CPUPRFLG(PNT) == 0)) THEN
            CPUPRFLG(PNT) = 1
            CP2FLG(NODE2) = 1
            IF(NODE1 == MYNODP) THEN
              PARTNER(NROUND) = NODE2
            ENDIF
            NPAIR = NPAIR + 1
            HIT = .TRUE.  !there is a hit for node1
            CLEAN = 1
           ENDIF !if flag2 and pair flag are 0
           PNT = PNT + 1
          ENDDO !loop 
        ENDDO !loop over all cpus
        IF(CLEAN == 0) DONE = .TRUE.
       ENDDO !loop while not done
!       WRITE(6,*) 'PASS ',PASSES,' MATCHED ',NPAIR,' CPU PAIRS '
!
       NROUND = NROUND -1 !we've overcounted rounds
!       IF(MYNODP == 1) THEN
!       WRITE(6,*) 'PASS ',PASSES,' THERE ARE ',NROUND,' ROUNDS '
!       DO ROUND = 1,NROUND
!         WRITE(6,*) 'MYNODP ',MYNODP,' ROUND ',ROUND,' PARTNER ',
!     & PARTNER(ROUND)  
!       ENDDO
!       ENDIF
! 
! now determine which atoms and clusters I have to send out--
! i.e. only the stuff in my cpu region
!
! note that that all work in this routine renumbers the cpu's such that
! cpu_here = cpu + 1
!      GOTO 7654
!       MYATCNT = 0
!       SCLSCNT = 0
!       LOCCLS = 0
!        LOCATM = 0 !atom counter for each cpu
!        WRITE(6,*) 'PASS ',PASSES,' MYNODE ',MYNOD,' START ',
!     & ISTART(MYNODP),' END ',IEND(MYNODP)
!        DO DUMCUB = ISTART(MYNODP),IEND(MYNODP) !loop over indices
!         ZER = SFCUBE(DUMCUB)
!         ACCN = 1
!         ACC = LSTM0(ZER)
!         DO WHILE(ACCN <= CNTN(ZER))
!          LOCACFLG(ACC) = 1  !flag the clusters in this cpu
!          CPUACMAP(ACC) = MYNOD 
!          SCLSCNT = SCLSCNT + 1
!          SCLSMAP(SCLSCNT) = ACC
!          LOCCLS = LOCCLS + 1
!          SCLUSX(SCLSCNT) = MEANX(ACC)
!          SCLUSY(SCLSCNT) = MEANY(ACC)
!          SCLUSZ(SCLSCNT) = MEANZ(ACC)
!          IQQ=ACLHI(ACC)
!          ISS=IQQ - ACLPTN(ACC) + 1
!          IF (ACLPTN(ACC) <= 0) CALL DIE
!          DO BBB = ISS,IQQ
!           MYATCNT = MYATCNT + 1
!           LOCATM = LOCATM + 1
!           ATM = JACLS(BBB)
!           MYATMAP(MYATCNT) = ATM
!           CPUMAP(ATM) = MYNOD   !did not add 1 
!           ICPUMAP(ATM) = MYNOD
!           MYATMX(MYATCNT) = X(ATM) 
!           MYATMY(MYATCNT) = Y(ATM)
!           MYATMZ(MYATCNT) = Z(ATM)
!          ENDDO
!          ACC = LSTN0(ACC)
!          ACCN = ACCN + 1
!         ENDDO !loop over clusters in primary cubes
!        ENDDO !loop over cubes in my cpu
! 7654   CONTINUE
!        GOTO 7655
!        IF(MYNODP == 4) THEN
!         DO AC1 = 1,NACTC
!          WRITE(6,*) 'CLUSTER  ',AC1,' LOCACFLG ',LOCACFLG(AC1)
!         ENDDO
!        CALL PRCPUMAP(ICPUMAP,NATOM,PASSES,MYNODP)
!        ENDIF
! 7655   CONTINUE
!
!        TESTNODE = 0 
!        WRITE(6,*) 'PASS ',PASSES,
!     & 'number of atoms in pcpu ',MYNODP,' = ',LOCATM
!        WRITE(6,*) 'PASS ',PASSES,
!     & 'number of clusters in pcpu ',MYNODP,' = ',LOCCLS
!        WRITE(6,*) 'PASS ',PASSES,' MYNODE ',MYNOD,' NROUND ',NROUND
!       GOTO 9876
!      GOTO 9432
!      WRITE(6,*) 'MYNODP ',MYNODP,' AFTER CALC ATOM ',
!     & AT1,' X = ',X(AT1),' PASS ',PASSES
!            WRITE(6,*) 'MYNODP ',MYNODP,' AFTER CALC ATOM ',
!     & AT1,' Y = ',Y(AT1),' PASS ',PASSES
!            WRITE(6,*) 'MYNODP ',MYNODP,' AFTER CALC ATOM ',
!     & AT1,' Z = ',Z(AT1),' PASS ',PASSES
! 9432  CONTINUE
!      ENDDO
!      WRITE(6,*) 'MYNODP ',MYNODP,' WEVE ERASED ',
!     & NZERO,'  ATOMS '
!
!        YRATCNT = 0
!        CALL SPACSRARG(NROUND,PARTNER,X,Y,Z,MYATCNT,
!     & MYATMAP,MYATMX,MYATMY,MYATMZ,YRATCNT,YRATMAP,YRATMX,
!     & YRATMY,YRATMZ)
!        CALL SPACSR(X,Y,Z)
!        GOTO 987
!        WRITE(6,*) 'CALLING SPACSRPACKA, NATOM IS ',NATOM
!        CALL SPACSRPACKA(NROUND,PARTNER,X,Y,Z,
!     & PRTATMLST,PRTATMHI,PRTATMMNY,NPARTATM,
!     & MYATMAP,MYATMX,MYATMY,MYATMZ,YRATMAP,YRATMX,
!     & YRATMY,YRATMZ,NATOM,PASSES)
! 987  CONTINUE
       CALL SPACSRSET(X,Y,Z)
!       CALL SPACSR(X,Y,Z)
!       CALL FORCSR(DX,DY,DZ)
!
!       DO AC1 = 1,NACTC
!          IQ=ACLHI(AC1)
!          IS=IQ - ACLPTN(AC1) + 1
!          IF (ACLPTN(AC1) <= 0) CALL DIE
!          DO BB = IS,IQ
!            AT1 = JACLS(BB)
!            IF (X(AT1) == 1000)
!     & WRITE(6,*) 'PASS ',PASSES,' MYNODP ',MYNODP,
!     & 'ATOM ',AT1,' X COORDINATE NOT COMMUNICATED',
!     & ' CLUSTER ',AC1
!            IF (Y(AT1) == 1000)
!     & WRITE(6,*) 'ATOM ',AT1,' Y COORDINATE NOT COMMUNICATED',
!     &  ' CLUSTER ',AC1
!            IF (Z(AT1) == 1000)
!     & WRITE(6,*) 'ATOM ',AT1,' Z COORDINATE NOT COMMUNICATED',
!     &  ' CLUSTER ',AC1
!          ENDDO
!        ENDDO
! now do clusters
!
!        RCLSCNT = 0
!        CALL SPACSRARG(NROUND,PARTNER,MEANX,MEANY,MEANZ,SCLSCNT,
!     & SCLSMAP,SCLUSX,SCLUSY,SCLUSZ,RCLSCNT,RCLSMAP,RCLUSX,
!     & RCLUSY,RCLUSZ)
!         CALL SPACSRPACKA(NROUND,PARTNER,MEANX,MEANY,MEANZ,
!     & PRTACCLST,PRTACCHI,PRTACCMNY,NPARTACC,
!     & SCLSMAP,SCLUSX,SCLUSY,SCLUSZ,RCLSMAP,RCLUSX,
!     & RCLUSY,RCLUSZ)
#endif /* (spacdec80)*/
!
!  Loop over cubes and in each cube, test each cluster.
!  For a given "test" cluster, loop over the set of surrounding (filled)
!  cubes, and for each cube, compare the position of each cluster
!  within the cube to that of the test cluster.  Save a cluster pair
!  only if the intercluster distance is less than (the non-
!  bonded cutoff + the "half-margins" of both clusters). Exclude a
!  cluster pair if the second cluster number < = the first.

       NCPNER = 0
       NCPDIS = 0
#if KEY_SPACDEC==1
       NMYACCL = 0 
#endif
!        
#if KEY_PARALLEL==1 /*parapenult*/
#if KEY_SPACDEC==1 /*spacdec90*/
       DO AC1 = 1,NACTC
        ATLDCLS(AC1) = 0  !initialize atom loads
! for spacdec: carry out the calculation only for my clusters
        IF(LOCACFLG(AC1) == 1) THEN
          NMYACCL = NMYACCL + 1
          MYACCL(NMYACCL)=AC1
#else /* (spacdec90)*/
       DO AC1 = MYNODP,NACTC,NUMNOD
#endif /* (spacdec90)*/
#else /* (parapenult)*/
       DO AC1 = 1,NACTC
#endif /* (parapenult)*/
        ZER = ORDCBN(AC1)
        PARTNN(AC1) = 0
        PARTND(AC1) = 0
        MAC1X = MEANX(AC1)
        MAC1Y = MEANY(AC1)
        MAC1Z = MEANZ(AC1)
        UU = HIMPRTN(ZER)
        SS = UU - NUMMP(ZER) + 1
        DO WHILE(SS <= UU)
          M = WRKAR(SS)
          J = 1
          AC2 = LSTM0(M)
          DO WHILE ((J <= CNTN(M)).AND.(AC2 > AC1))
            IMAR = CUTNB - HAFMAR(AC2) - HAFMAR(AC1)
            IF(IMAR < 0) THEN
             IMARSQ = 0
            ELSE
             IMARSQ = IMAR*IMAR
            ENDIF
            OMARG = (HAFMAR(AC2) + HAFMAR(AC1) &
       + CUTNB)
            MARGSQ = OMARG*OMARG
            XM = MEANX(AC2) - MAC1X
            YM = MEANY(AC2) - MAC1Y
            ZM = MEANZ(AC2) - MAC1Z
            DSQ = XM*XM+YM*YM+ZM*ZM
            IF (DSQ  <=  IMARSQ) THEN
              NCPNER = NCPNER + 1
              CPNER(NCPNER) = AC2
              PARTNN(AC1) = PARTNN(AC1) + 1
            ELSE IF (DSQ  <=  MARGSQ) THEN
              NCPDIS = NCPDIS + 1
              CPDIS(NCPDIS) = AC2
              PARTND(AC1) = PARTND(AC1) + 1
            ENDIF
           AC2 = LSTN0(AC2)
           J = J + 1
          ENDDO
        SS = SS +1
        ENDDO
        IF (((NCPDIS + NACTC) > MAXACD).OR. &
       ((NCPNER + NACTC) > MAXACN)) THEN
          IF ((NCPDIS + NACTC) > MAXACD) THEN
           NCPDIS = MAXACD + 1
#if KEY_PARALLEL==1 /*maxacdsze*/
        IF(NUMNOD > 1) THEN
        WRITE(OUTU,'(A)') &
       ' DISTANT CLUSTER ARRAY SIZE EXCEEDED; INCREASE NBSCALE'
        CALL WRNDIE(-5,'<NBNDCCO>','ARRAY SIZE LIMIT EXCEEDED')
        ENDIF
#endif /* (maxacdsze)*/
           RETURN
          ENDIF
          IF ((NCPNER + NACTC) > MAXACN) THEN
           NCPNER = MAXACN + 1
#if KEY_PARALLEL==1 /*maxacnsze*/
        IF(NUMNOD > 1) THEN
        WRITE(OUTU,'(A)') &
       ' NEAR CLUSTER ARRAY SIZE EXCEEDED; INCREASE NBSCALE'
        CALL WRNDIE(-5,'<NBNDCCO>','ARRAY SIZE LIMIT EXCEEDED')
        ENDIF
#endif /* (maxacnsze)*/
           RETURN
          ENDIF
        ENDIF
       PNTNER(AC1) = NCPNER
       PNTDIS(AC1) = NCPDIS
#if KEY_SPACDEC==1
       ENDIF  /*(locacflg)*/
#endif
       ENDDO
       IF (PRNLEV > 3) THEN
         WRITE(OUTU,'(A,I8)')'TOTAL NEAR CLUSTER PAIRS',NCPNER
         WRITE(OUTU,'(A,I8)')'TOTAL DISTANT CLUS PAIRS',NCPDIS
       ENDIF
!
! -----------------------------------------------------
! The final loop, which does the final atom-atom distance
! calculations and exclusions, and generates the non-bonded
! list.
!
       NNNB = 0
!
!  First loop over all clusters.  For each cluster
!  loop over all the atoms it contains.
#if KEY_PARALLEL==1 /*parafinal*/
#if KEY_SPACDEC==1 /*spacdec100*/
!       DO AC1 = 1,NACTC
        DO ACC = 1,NMYACCL
         AC1 = MYACCL(ACC)
#else /* (spacdec100)*/
       DO AC1 = MYNODP,NACTC,NUMNOD
#endif /* (spacdec100)*/
#else /* (parafinal)*/
       DO AC1 = 1,NACTC
#endif /* (parafinal)*/
!        WRITE(6,*) 'MYNODP ',MYNODP,' CLUSTER ',AC1,' LOCACFLG ',
!     & LOCACFLG(AC1),' PASS ',PASSES  !temporary
#if KEY_SPACDEC==1
!        IF(LOCACFLG(AC1) == 1) THEN  
#endif
        IQ=ACLHI(AC1)
        IS=IQ - ACLPTN(AC1) + 1
        IF (ACLPTN(AC1) <= 0) CALL DIE
        DO BB = IS,IQ
          AT1 = JACLS(BB)
          MAXDIF = NUEXCA(AT1)
! store the atom1-dependent part of the
! exclusion (table) array pointers and the
! coordinates of atom1
          B1 = ATEXPT(AT1)
          FIXED = IMOVE(AT1) > 0
          X1 = X(AT1)
          Y1 = Y(AT1)
          Z1 = Z(AT1)
!  Do inTRAcluster atom comparisons -------------------
! Added line below--RJP 7.20.99
          STRTAT=BB+1
          DO DD = STRTAT,IQ
           AT2 = JACLS(DD)
           DOIT = .TRUE.
#if KEY_TSM==1 /*tsmatom*/
           IF(LTSM) THEN                             
             IF(REACLS(AT1) == 1.AND.PRODLS(AT2).EQ.1.OR. &
             REACLS(AT2) == 1 .AND.PRODLS(AT1).EQ.1) DOIT=.FALSE.
           ENDIF                                     
#endif /* (tsmatom)*/
           IF(FIXED.AND.(IMOVE(AT2) > 0)) DOIT = .FALSE.
           IF(DOIT) THEN
             XX1 = X(AT2) - X1
             YY1 = Y(AT2) - Y1
             ZZ1 = Z(AT2) - Z1
             IF (XX1*XX1 + YY1*YY1 + ZZ1*ZZ1  <=  CTNBSQ) THEN
!   check exclusions for this pair:
              DIFF = AT2-AT1
              IF (DIFF > MAXDIF) THEN
               NNNB = NNNB + 1
               JNB(NNNB) = AT2
#if KEY_SPACDEC==1
               ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
              ELSE
               IF (EXCLAR(B1 + DIFF) == 1) THEN
                NNNB = NNNB + 1
                JNB(NNNB) = AT2
#if KEY_SPACDEC==1
                ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
               ELSE IF (EXCLAR(B1 + DIFF) == -1) THEN
                NNNB = NNNB + 1
                JNB(NNNB) =-AT2
#if KEY_SPACDEC==1
                ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
               ENDIF
              ENDIF !if possibly an exclusion
             ENDIF !if less than cutoff
           ENDIF !if distance testing is appropriate (doit)
          ENDDO
!  Do nearby inTERcluster comparisons (not requiring
! atom-atom distance calculations) ---------------------------------
          TT = PNTNER(AC1) - PARTNN(AC1) + 1
          DO CC = TT,PNTNER(AC1)
           AC2 = CPNER(CC)
           IQ2 =ACLHI(AC2)
           IS2=IQ2 - ACLPTN(AC2) + 1
           IF (ACLPTN(AC2) <= 0) CALL DIE
           DO DD = IS2,IQ2
            AT2 = JACLS(DD)
            DOIT = .TRUE.
#if KEY_TSM==1 /*tsmatom*/
            IF(LTSM) THEN                             
             IF(REACLS(AT1) == 1 .AND.PRODLS(AT2).EQ.1.OR. &
             REACLS(AT2) == 1 .AND.PRODLS(AT1).EQ.1) DOIT=.FALSE.
            ENDIF                                     
#endif /* (tsmatom)*/
            IF(FIXED.AND.(IMOVE(AT2) > 0)) DOIT = .FALSE.
            IF(DOIT) THEN
             DIFF = AT2-AT1
             IF (DIFF > MAXDIF) THEN
               NNNB = NNNB + 1
               JNB(NNNB) = AT2
#if KEY_SPACDEC==1
               ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
             ELSE
               IF (EXCLAR(B1 + DIFF) == 1) THEN
                NNNB = NNNB + 1
                JNB(NNNB) = AT2
#if KEY_SPACDEC==1
                ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
               ELSE IF (EXCLAR(B1 + DIFF) == -1) THEN
                NNNB = NNNB + 1
                JNB(NNNB) =-AT2
#if KEY_SPACDEC==1
                ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
               ENDIF
             ENDIF
            ENDIF !if dist testing is appropriate (doit)
           ENDDO
          ENDDO
! Do distant inTERcluster comparisons (requiring
! atom-atom distance calculations) --------------------------------------
          TT = PNTDIS(AC1) - PARTND(AC1) + 1
          DO CC = TT,PNTDIS(AC1)
           AC2 = CPDIS(CC)
           IQ2 =ACLHI(AC2)
           IS2=IQ2 - ACLPTN(AC2) + 1
           IF (ACLPTN(AC2) <= 0) CALL DIE
           DO DD = IS2,IQ2
            AT2 = JACLS(DD)
            DOIT = .TRUE.
#if KEY_TSM==1 /*tsmatom*/
            IF(LTSM) THEN                       
             IF(REACLS(AT1) == 1 .AND.PRODLS(AT2).EQ.1.OR. &
             REACLS(AT2) == 1 .AND.PRODLS(AT1).EQ.1) DOIT=.FALSE.
            ENDIF         
#endif /* (tsmatom)*/
            IF(FIXED.AND.(IMOVE(AT2) > 0)) DOIT = .FALSE.
            IF(DOIT) THEN
              XX1 = X(AT2) - X1
              YY1 = Y(AT2) - Y1
              ZZ1 = Z(AT2) - Z1
              IF (XX1*XX1 + YY1*YY1 + ZZ1*ZZ1  <=  CTNBSQ) THEN
!   check exclusions for this pair:  ----------------------
               DIFF = AT2-AT1
               IF (DIFF > MAXDIF) THEN
                NNNB = NNNB + 1
                JNB(NNNB) = AT2
#if KEY_SPACDEC==1
                ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
               ELSE
                IF (EXCLAR(B1 + DIFF) == 1) THEN
                 NNNB = NNNB + 1
                 JNB(NNNB) = AT2
#if KEY_SPACDEC==1
                 ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
                ELSE IF (EXCLAR(B1 + DIFF) == -1) THEN
                 NNNB = NNNB + 1
                 JNB(NNNB) =-AT2
#if KEY_SPACDEC==1
                 ATLDCLS(AC1) = ATLDCLS(AC1) + 1  
#endif
                ENDIF
               ENDIF
              ENDIF
            ENDIF !if dist testing is appropriate (doit)
           ENDDO
          ENDDO
          INBLO(AT1) = NNNB
          IF ((NNNB + NACTVE) > MAXJNB) THEN
#if KEY_PARALLEL==1 /*maxjnbsze*/
        IF(NUMNOD > 1) THEN
        WRITE(OUTU,'(A)') &
       ' NON BOND LIST ARRAY SIZE EXCEEDED; INCREASE NBSCALE'
        CALL WRNDIE(-5,'<NBNDCCO>','ARRAY SIZE LIMIT EXCEEDED')
        ENDIF
#endif /* (maxjnbsze)*/
           NNNB = MAXJNB + 1
           RETURN
          ENDIF
        ENDDO !loop over atoms in primary clusters
!        AC1 = LSTN0(AC1)
!        AC1N = AC1N + 1
#if KEY_SPACDEC==1
!        ENDIF  /*if loc ac flag  >  1*/
#endif
       ENDDO  !loop over clusters
!      ENDDO  !loop over cubes
!
! since the non-bonded list has been filled only for active atoms
! (rest of INBLO(I) are zero) fill in the rest as appropriate --------
!
      DO I = 2,NATOM
       IF (INBLO(I) == 0) INBLO(I) = INBLO(I-1)
!        WRITE(6,*) 'MYNODP ', MYNODP,' ATOM ',I,' INBLO ',INBLO(I)
      ENDDO
!----------------------------------------------------------------------
!---- Flag success and print statistics -------------------------------
!
      CMPLTD=.TRUE.
!
!      IF (PRNLEV >= 2)
!     $     WRITE(OUTU,*)'MYNODP ',MYNODP,  
!     & ' NBNDCCO found: ', NNNB, ' atom pairs'
       
       IF(PRNLEV >= 2) WRITE(OUTU,*) &
       ' NBNDCCO found: ', NNNB, ' atom pairs'
!
      PRNLEV = PLVOLD
!
#if KEY_SPACDEC==1
      LSPACFRST = .FALSE.  
#endif
#endif /* (yesbycc)*/
      RETURN
      END

      SUBROUTINE NBNDGG(X,Y,Z, &
           NNNBG,MXJNBG,JNBG,INBLOG,ING14,IGLO14, &
           CUTNB,CTEXNB,LEXTND, &
           LQUAD,LGRAD,WRNMIN,CMPLTD,EPS,MXNCUB, &
           LSTN0,XOO,YOO,ZOO,ORDCBN,NUMMP,HIMPRTN, &
           MMM,WRKAR,LSTM0,CNTN,XO,YO,ZO, &
           EXCLARG,GREXPT,NUEXCG, &
           RSCMX,RSCMY,RSCMZ,RSQ, &
           RSDX,RSDY,RSDZ,RSQXX,RSQYY,RSQZZ, &
           RSQXY,RSQYZ,RSQZX,RSXMAX,RSYMAX,RSZMAX,RSDISP, &
           RSPOT,RSFX,RSFY,RSFZ, &
           RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
           ATSX,ATSY,ATSZ,ATPOT,ATFX,ATFY,ATFZ, &
           ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX)
#if KEY_NO_BYCC==0 /*yesbycc2*/
!
  use chm_kinds
  use dimens_fcm
  use contrl
  use stream
  use timerm
  use consta
  use corman3,only:squareup
  use actclus_mod
  use psf
  use number
  use parallel
  use machutil,only:die
  use new_timer,only: seconds 
      implicit none
!
! This routine generates the group-group particle
! list for group-based energy calculations.  It is
! similar in approach to the atom-atom routine
! NBNDCC, but it uses chemical groups instead of
! clusters and of course does not carry out atom-atom
! comparisons.
! When extended electrostatics are requested, the cube
! size in the compartmentalization is based on the
! extended cut-off distance (CTEXNB), not on the non-
! bonded cut-off distance (CUTNB), since all group-group
! pairs at a distance of CTEXNB need ultimately to be
! specified.  Hence the speed of the algorithm will
! decrease for large CTEXNB when electrostatics is
! requested.
! Note that the speed of the algorithm, however,
! does not depend on the size of the groups, since
! they are essentially treated as points in space
! (unlike the clusters of NBNDCC).
! The extended electrostatics are modified from
! nbondg.src, by Bernard R. Brooks.
!
!                             -RJ Petrella, 1.6.99
!
! Passed variables
!
      real(chm_real) X(*),Y(*),Z(*)
      INTEGER NNNBG,MXJNBG
      INTEGER JNBG(*)
      INTEGER INBLOG(*)
      INTEGER IGLO14(*),ING14(*)
      real(chm_real) CUTNB,CTEXNB
      LOGICAL LEXTND,LQUAD,LGRAD
      real(chm_real) WRNMIN
      LOGICAL CMPLTD,QMOVE
      real(chm_real) EPS
      real(chm_real) ATSX(*),ATSY(*),ATSZ(*),ATPOT(*), &
           ATFX(*),ATFY(*),ATFZ(*)
      real(chm_real) ATGXX(*),ATGYY(*),ATGZZ(*), &
           ATGXY(*),ATGYZ(*),ATGZX(*)
      INTEGER MXNCUB
      INTEGER EXCLARG(*),GREXPT(*),NUEXCG(*)
!
! **********************************************************************
! "LOCAL" ARRAYS PASSED FOR MEMORY ALLOCATION ONLY
!
      INTEGER LSTN0(*),XOO(*),YOO(*),ZOO(*),ORDCBN(*)
      real(chm_real) RSCMX(*),RSCMY(*),RSCMZ(*),RSQ(*)
      real(chm_real) RSDX(*),RSDY(*),RSDZ(*),RSQXX(*),RSQYY(*),RSQZZ(*)
      real(chm_real) RSQXY(*),RSQYZ(*),RSQZX(*), &
           RSXMAX(*),RSYMAX(*),RSZMAX(*)
      INTEGER RSDISP(*)
      real(chm_real) RSPOT(*),RSFX(*),RSFY(*),RSFZ(*)
      real(chm_real) RSGXX(*),RSGYY(*),RSGZZ(*), &
           RSGXY(*),RSGYZ(*),RSGZX(*)
!  Cube arrays
      INTEGER XO(*),YO(*),ZO(*) !grid coors of cubes
      INTEGER LSTM0(*),CNTN(*),NUMMP(*)
      INTEGER HIMPRTN(*),MMM(*),WRKAR(*)
! **********************************************************************
!  Atom arrays
      real(chm_real) MAXX(3),MINX(3),MAXY(3),MINY(3),MAXZ(3),MINZ(3) !coor extrema
! **********************************************************************
! LOCAL SCALARS
!   for exclusions
      INTEGER DIFF,MAXDIFG,B1,B2,NXI,NXIMAX
      INTEGER EXCL14,LEX14,ELEN
!   for cubes
      INTEGER TOTCUB,CUBMAX
      real(chm_real) CUBNEED,CUBL,CUTT
      real(chm_real) XR,YR,ZR
      INTEGER MNUM,PP,FLDM
!  for groups
      INTEGER ACC,AC1,AC2
      INTEGER ACC1,AC,ACG,ACG1,ACG2
      INTEGER I,ZER,J,KK,M,CC,UU,SS,QQ
      INTEGER NUMBX,NUMBY,NUMBZ,NXY
      real(chm_real) XAVG,YAVG,ZAVG,XSUM,YSUM,ZSUM
      LOGICAL FIXED
      real(chm_real) MAC1X,MAC1Y,MAC1Z
      real(chm_real) DIFX,DIFY,DIFZ,DISTSQ,MNCUTF
      real(chm_real) CTNBSQ,CTEXSQ,WARNDIS
! for extended electrostatics
      INTEGER IGPBD(4)
      INTEGER JS,JQ,IRST,JRST
      INTEGER IS,IQ,NAT,NGPE,NGPX,NGPN
      real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,XD,YD,ZD
      real(chm_real) R2,R,XI,YI,ZI
      real(chm_real) QSUM,DXT,DYT,DZT
      real(chm_real) QXXT,QYYT,QZZT,QXYT,QZXT,QYZT,CGT
      real(chm_real) XY,YZ,ZX,XX,YY,ZZ
      real(chm_real) R1,R3,R5,R2X3,R2X5,R2X7,DOT,QXT,QYT,QZT,RQR,CR3
      real(chm_real) TEMP,TEMP2
      PARAMETER(WARNDIS = 8) ! excl group pair warning distance
! other
      INTEGER PLVOLD
      real(chm_real) :: time1,etime,ctime,cumetime_cc=0 
      logical :: QTIMING=.false.
!
!   End of variables for O(N) search
!
!---- Initial housekeeping --------------------------------------------
!
   if(QTIMING) then
    call seconds(etime,ctime)
    time1 = etime
   endif
!     store initial atom positions
      DO I=1,NATOM
         ATSX(I)=X(I)
         ATSY(I)=Y(I)
         ATSZ(I)=Z(I)
      ENDDO
!
      PLVOLD = PRNLEV
      IF (DYNAMQ.OR.MINXYZ) PRNLEV = 3
      IF(PRNLEV > 3) WRITE(OUTU,'(A)') &
       'Using C-IN-C search for group pairs'
!
      IF (NACTG == 0) CALL WRNDIE(-5,'<NBNDGG>', &
        'NO ACTIVE GROUPS DEFINED')
      CMPLTD=.FALSE.            ! Will remain .FALSE. until very end
      IF (LEXTND) THEN
       CUTT = CTEXNB
      ELSE
       CUTT = CUTNB
      ENDIF
      CTNBSQ=CUTNB*CUTNB
      CTEXSQ=CTEXNB*CTEXNB
!  Initialize non-bonded list --necessary since unassigned elements
! MUST equal zero later in algorithm.
      DO I = 1,NGRP
        INBLOG(I)=ZERO
      ENDDO
!
!   Zero out counters for groups of various charge types
      DO I = 1,4
         IGPBD(I)=0
      ENDDO
      NGPN = 0
      NGPE = 0
      NGPX = 0
!
!---- Initialize extended electrostatics ------------------------------
!
!   Store the starting atom positions and initialize atom potentials
!
!   The loop on I is vectorizable.
!
      IF (LEXTND) THEN
! Intialize atom potentials.
         DO I=1,NATOM
            ATPOT(I)=0.0
            ATFX(I)=0.0
            ATFY(I)=0.0
            ATFZ(I)=0.0
            IF (LGRAD) THEN
               ATGXX(I)=0.0
               ATGYY(I)=0.0
               ATGZZ(I)=0.0
               ATGXY(I)=0.0
               ATGYZ(I)=0.0
               ATGZX(I)=0.0
            ENDIF
! Store the current atom configuration.
            ATSX(I)=X(I)
            ATSY(I)=Y(I)
            ATSZ(I)=Z(I)
         ENDDO
!
       DO I = 1,NGRP
!
            RSPOT(I)=ZERO
            RSFX(I)=ZERO
            RSFY(I)=ZERO
            RSFZ(I)=ZERO
            RSGXX(I)=ZERO
            RSGYY(I)=ZERO
            RSGZZ(I)=ZERO
            RSGXY(I)=ZERO
            RSGYZ(I)=ZERO
            RSGZX(I)=ZERO
         ENDDO
       END IF
!
!---- END initialize extended electrostatics ------------------------------
!
!   FIND GEOMETRIC CENTER FOR EACH RESIDUE AND THE MULTIPOLE MOMENTS
!   FOR THE CHARGE DISTRIBUTIONS
!
!.... Loop over groups I begins here ..................................
!
!
      DO I = 1,NACTG
       ACG = ACTVG(I)
!
!....... Increment counter for group charge type ......................
!
         J=IGPTYP(ACG)+1
         IGPBD(J)=IGPBD(J)+1
!
!....... Find geometric center and bounding box .......................
!
         IS=IGPBS(ACG)+1
         IQ=IGPBS(ACG+1)
         NAT=IQ-IS+1
         IF (NAT <= 0) CALL DIE
         XMIN=X(IS)
         XMAX=XMIN
         YMIN=Y(IS)
         YMAX=YMIN
         ZMIN=Z(IS)
         ZMAX=ZMIN
         DXT=ZERO
         DYT=ZERO
         DZT=ZERO
         DO J = IS,IQ
            DXT=DXT+X(J)
            DYT=DYT+Y(J)
            DZT=DZT+Z(J)
            XMIN = MIN(XMIN,X(J))
            YMIN = MIN(YMIN,Y(J))
            ZMIN = MIN(ZMIN,Z(J))
            XMAX = MAX(XMAX,X(J))
            YMAX = MAX(YMAX,Y(J))
            ZMAX = MAX(ZMAX,Z(J))
         END DO
         XD=DXT/NAT
         YD=DYT/NAT
         ZD=DZT/NAT
! Store positions of centers of groups
         RSCMX(ACG)=XD
         RSCMY(ACG)=YD
         RSCMZ(ACG)=ZD
!
!....... Find smallest bounding box centered on geometric center ......
!
         RSXMAX(ACG)=MAX(XMAX-XD,XD-XMIN)
         RSYMAX(ACG)=MAX(YMAX-YD,YD-YMIN)
         RSZMAX(ACG)=MAX(ZMAX-ZD,ZD-ZMIN)
!
!
!....... Compute multipole moments for this group .....................
!
         IF (LEXTND) THEN
            DXT=ZERO
            DYT=ZERO
            DZT=ZERO
            QSUM=ZERO
            QXXT=ZERO
            QYYT=ZERO
            QZZT=ZERO
            QXYT=ZERO
            QZXT=ZERO
            QYZT=ZERO
            DO J = IS,IQ
               CGT=CCELEC*CG(J)/EPS
               QSUM=QSUM+CGT
               DXT=DXT+CGT*X(J)
               DYT=DYT+CGT*Y(J)
               DZT=DZT+CGT*Z(J)
               QXXT=QXXT+CGT*X(J)*X(J)
               QYYT=QYYT+CGT*Y(J)*Y(J)
               QZZT=QZZT+CGT*Z(J)*Z(J)
               QXYT=QXYT+CGT*X(J)*Y(J)
               QZXT=QZXT+CGT*X(J)*Z(J)
               QYZT=QYZT+CGT*Y(J)*Z(J)
            ENDDO
            RSQ(ACG)=QSUM
            RSDX(ACG)=DXT-QSUM*XD
            RSDY(ACG)=DYT-QSUM*YD
            RSDZ(ACG)=DZT-QSUM*ZD
            IF (LQUAD) THEN
               QXXT=QXXT+QSUM*XD*XD-2.0*DXT*XD
               QYYT=QYYT+QSUM*YD*YD-2.0*DYT*YD
               QZZT=QZZT+QSUM*ZD*ZD-2.0*DZT*ZD
               RSQXX(ACG)=2.0*QXXT-QYYT-QZZT
               RSQYY(ACG)=2.0*QYYT-QXXT-QZZT
               RSQZZ(ACG)=2.0*QZZT-QXXT-QYYT
               RSQXY(ACG)=3.0*(QXYT+QSUM*XD*YD-DXT*YD-DYT*XD)
               RSQYZ(ACG)=3.0*(QYZT+QSUM*YD*ZD-DYT*ZD-DZT*YD)
               RSQZX(ACG)=3.0*(QZXT+QSUM*XD*ZD-DXT*ZD-DZT*XD)
            END IF
         END IF
       END DO
! ********************************************************************
! ************Start search for group pairs****************************
! ********************************************************************
!
!  The minimum cube side length is the maximum distance between any
!  two groups in the group exclusion list, since ALL (active) excluded group
!  pairs must be included in the final group-group non-bonded pairlist.
!  If one or more excluded group-group distances are very large,
!  (> WARNDIS) a warning is issued.
!
      MNCUTF = 0
      DO I = 1,NACTG
        ACG1 = ACTVG(I)
        IF(ACG1 > 1) THEN
          NXI=IGLO14(ACG1-1)+1
        ELSE
          NXI=1
        ENDIF
        NXIMAX=IGLO14(ACG1)
        DO J = NXI,NXIMAX
          ACG2 = ING14(J)
          IF (ACGFLG(ACG2) == 1) THEN
           DIFX = RSCMX(ACG1)-RSCMX(ACG2)
           DIFX = DIFX*DIFX
           DIFY = RSCMY(ACG1)-RSCMY(ACG2)
           DIFY = DIFY*DIFY
           DIFZ = RSCMZ(ACG1)-RSCMZ(ACG2)
           DIFZ = DIFZ*DIFZ
           DISTSQ = DIFX + DIFY + DIFZ
           IF (DISTSQ > MNCUTF) MNCUTF = DISTSQ
          ENDIF
        ENDDO
      ENDDO
!
      MNCUTF = SQRT(MNCUTF)
      IF (PRNLEV > 3) WRITE (OUTU,'(A,F7.3)') &
       'MIN NECESSARY CUBE LENGTH ',MNCUTF
      IF ((PRNLEV > 2).AND.(MNCUTF.GT.WARNDIS)) &
       CALL WRNDIE(0,'<NBNDGG>', &
       'NBNDGG> LARGE GRP-GRP EXCL PAIR DISTANCE')
!
!---- Zero out result counter ----------------------------------------
      NNNBG = 0
!
! set cube length equal to whatever
!  specified above: either ctexnb or cutnb
      CUBL = CUTT
! make sure cube length is at least as large as the maximum
! group-group distance in the group exclusion list
      IF (CUBL < MNCUTF) CUBL = MNCUTF
      IF (PRNLEV > 3) WRITE (OUTU,'(A,F7.3)') &
       'CUBE LENGTH ',CUBL
! initialize sums for center of geometry calculation
! & initialize coordinate limits
      XSUM = 0
      YSUM = 0
      ZSUM = 0
       DO KK = 1,3
        MINX(KK) = 99999
        MAXX(KK) = -99999
        MINY(KK) = 99999
        MAXY(KK) = -99999
        MINZ(KK) = 99999
        MAXZ(KK) = -99999
       ENDDO
! -------- Loop over groups to find
! center of geometry and coordinate limits
! (in terms of whole groups-- different from section
! immediately above, which does this in term of atoms)
! ----------------------------------------------------
      DO I = 1,NACTG
        ACC = ACTVG(I)
        XSUM = XSUM + RSCMX(ACC)
        YSUM = YSUM + RSCMY(ACC)
        ZSUM = ZSUM + RSCMZ(ACC)
        IF (RSCMX(ACC) > MAXX(1)) THEN
           MAXX(1) = RSCMX(ACC)
           MAXX(2) = RSCMY(ACC)
           MAXX(3) = RSCMZ(ACC)
        ENDIF
        IF (RSCMX(ACC) < MINX(1)) THEN
           MINX(1) = RSCMX(ACC)
           MINX(2) = RSCMY(ACC)
           MINX(3) = RSCMZ(ACC)
        ENDIF
        IF (RSCMY(ACC) > MAXY(2)) THEN
           MAXY(1) = RSCMX(ACC)
           MAXY(2) = RSCMY(ACC)
           MAXY(3) = RSCMZ(ACC)
        ENDIF
        IF (RSCMY(ACC) < MINY(2)) THEN
           MINY(1) = RSCMX(ACC)
           MINY(2) = RSCMY(ACC)
           MINY(3) = RSCMZ(ACC)
        ENDIF
        IF (RSCMZ(ACC) > MAXZ(3)) THEN
           MAXZ(1) = RSCMX(ACC)
           MAXZ(2) = RSCMY(ACC)
           MAXZ(3) = RSCMZ(ACC)
        ENDIF
        IF (RSCMZ(ACC) < MINZ(3)) THEN
           MINZ(1) = RSCMX(ACC)
           MINZ(2) = RSCMY(ACC)
           MINZ(3) = RSCMZ(ACC)
        ENDIF
      ENDDO
      XAVG = XSUM/NACTG
      YAVG = YSUM/NACTG
      ZAVG = ZSUM/NACTG
5     FORMAT(A,F8.2,A,F8.2)
      DO KK = 1,3
       IF ((MAXX(KK) >= 9999).OR.(MAXY(KK).GE.9999).OR. &
        (MAXZ(KK) >= 9999)) THEN
        CALL WRNDIE(-5,'<NBNDGG>', &
       'NBNDGG> SOME COORDINATES >= 9999 (UNDEFINED)')
       ENDIF
      ENDDO
      NUMBX = INT((MAXX(1)-MINX(1))/CUBL) + 1
      NUMBY = INT((MAXY(2)-MINY(2))/CUBL) + 1
      NUMBZ = INT((MAXZ(3)-MINZ(3))/CUBL) + 1
      TOTCUB = NUMBX*NUMBY*NUMBZ
      IF (TOTCUB > MXNCUB) THEN
       IF (PRNLEV > 3) THEN
         WRITE(OUTU,'(A)') 'TOTAL CUBES = ', TOTCUB, &
       'EXCEED LIM = ', MXNCUB
         WRITE(OUTU,'(A)') 'TRYING TO SQUARE UP STRUCTURE'
       ENDIF
14     FORMAT(A,I10)
15     FORMAT(A,F8.2)
       CALL SQUAREUP(MAXX,MINX,MAXY,MINY,MAXZ,MINZ,XAVG, &
                     YAVG,ZAVG,RSCMX,RSCMY,RSCMZ,NGRP,ACTVG,NACTG)
       XR = MAXX(1)-MINX(1)
       YR = MAXY(2)-MINY(2)
       ZR = MAXZ(3)-MINZ(3)
       NUMBX = INT(XR/CUBL) + 1
       NUMBY = INT(YR/CUBL) + 1
       NUMBZ = INT(ZR/CUBL) + 1
       TOTCUB = NUMBX*NUMBY*NUMBZ
       IF (PRNLEV > 3) THEN
         WRITE(OUTU,14) 'TOT CUBES (AFTER SQUARE) = ', &
         TOTCUB
       ENDIF
        IF (TOTCUB > MXNCUB) THEN
        CUBNEED = CUBL
        DO WHILE(TOTCUB > MXNCUB)
        CUBNEED = ((XR + CUBNEED)*(YR + CUBNEED)* &
        (ZR + CUBNEED))
         CUBNEED = ((CUBNEED/MXNCUB)**0.33333333)
         NUMBX = INT(XR/CUBNEED) + 1
         NUMBY = INT(YR/CUBNEED) + 1
         NUMBZ = INT(ZR/CUBNEED) + 1
         TOTCUB = NUMBX*NUMBY*NUMBZ
         WRITE(OUTU,'(A,F6.3,A,I9)') ' CNEED ', &
        CUBNEED,' TOTCUB ',TOTCUB
        ENDDO
        CUBL = CUBNEED
        IF (PRNLEV >= 2) THEN
          WRITE(OUTU,15) 'EXPANDING CUBL to CUBNEED= ', &
        CUBL
        ENDIF
       ENDIF
      ENDIF
!
      IF (PRNLEV > 3) THEN
         WRITE (OUTU,'(A)') &
            'NBONDG Building group interaction list using grid'
         WRITE (OUTU,*) 'Number of active groups        =', NACTG
         WRITE (OUTU,*) 'Number of cells in X dimension =', NUMBX
         WRITE (OUTU,*) 'Number of cells in Y dimension =', NUMBY
         WRITE (OUTU,*) 'Number of cells in Z dimension =', NUMBZ
         WRITE (OUTU,*) 'Number of cells, total         =', TOTCUB
      ENDIF
      IF (PRNLEV > 3) WRITE (OUTU,'(A,F7.2)') &
       ' Cell size                      =', CUBL
!
!
      NXY = NUMBX*NUMBY
! ---initialize some cube arrays
       DO I = 1, TOTCUB
        LSTM0(I) = 0
        CNTN(I) = 0
        NUMMP(I) = 0
       ENDDO
! check to see which cubes are "filled" (non-empty) and
! for the filled ones, store:
! their positions on the cubical grid (XO,YO,ZO)
! and the number of groups they contain (CNTN)
! Also, store the total number of filled cubes (FLDM)
       FLDM = 0
       DO I = 1,NACTG
        AC = ACTVG(I)
        LSTN0(AC) = 0
        XOO(AC) = INT((RSCMX(AC)-MINX(1))/CUBL)
        YOO(AC) = INT((RSCMY(AC)-MINY(2))/CUBL)
        ZOO(AC) = INT((RSCMZ(AC)-MINZ(3))/CUBL)
        ORDCBN(AC) = XOO(AC) + (YOO(AC))*NUMBX + &
       (ZOO(AC))*NXY + 1
        IF (ORDCBN(AC) > MXNCUB) THEN
         CALL WRNDIE(-5,'<NBNDGG>', &
       'NBNDGG> TOO MANY CUBES. INCRSE CUBL OR REDUCE SYSTEM')
        ENDIF
        M = ORDCBN(AC)
        IF (CNTN(M) == 0) THEN
         FLDM = FLDM + 1
         MMM(FLDM) = M
!  Note:  MMM not sorted
         ZO(M) = ZOO(AC)
         YO(M) = YOO(AC)
         XO(M) = XOO(AC)
        ENDIF
        CNTN(M) = CNTN(M) + 1
       ENDDO
! --------------------------------------------------------
!    From Tom Ngo's "intertwined lists" algorithm.
!    For each cube, the highest-numbered group contained
!    in the cube is stored (in LSTM0()).  This group is
!    also linked to the next highest group in number
!    contained in the cube, such that  LSTN0(high) =
!    next highest. This is done for all groups in a given
!    cube.  Hence , the groups within the same cube are
!    linked to each other in a sort of "chain" and
!    they are all linked to the cube number via
!    the high group number.
       DO I = 1, NACTG
        ACC1 = ACTVG(I)
        M = ORDCBN(ACC1)
         LSTN0(ACC1) = LSTM0(M)
         LSTM0(M) = ACC1
       ENDDO
! ----------------------------------------------------
!   For each filled cube, determine the set of filled cubes in its
!   immediate surroundings and store the indices of these
!   filled cubes in a linked list
       MNUM = 0
       DO CC = 1,FLDM
        ZER = MMM(CC)
        KK = 1
        DO WHILE(KK <= 27)
         IF(((BBX(KK)+XO(ZER)) >= 0).AND.((BBY(KK)+YO(ZER)).GE.0).AND. &
          ((BBZ(KK)+ZO(ZER)) >= 0)) THEN
!  Next line means the cube can't be out of bounds
          IF(((BBX(KK)+XO(ZER)) < NUMBX).AND. &
        ((BBY(KK)+YO(ZER)) < NUMBY) &
           .AND.((BBZ(KK)+ZO(ZER)) < NUMBZ)) THEN
           PP = BBX(KK) + BBY(KK)*NUMBX + BBZ(KK)*NXY + ZER
           IF (CNTN(PP) > 0) THEN
            MNUM = MNUM + 1
            NUMMP(ZER) = NUMMP(ZER) + 1
            WRKAR(MNUM) = PP
           ENDIF
          ENDIF
         ENDIF
        KK = KK + 1
        ENDDO
        HIMPRTN(ZER) = MNUM
       ENDDO
!
!  ----------------------------------------------------------------
!  For a given "test" group, loop over the set of surrounding (filled)
!  cubes, and for each cube, compare the position of each group
!  within the cube to that of the test group.  Save a group pair
!  only if the intergroup distance is less than the non-bonded
!  cut-off distance.
!  Exclude a group pair if the second group number is lower than
!  the first
!  Criteria for inclusion in group list:
!   If both groups are fixed, they are completely ignored.
!   Otherwise:
!    If group pair is an exclusion, always included in group list as (-)
!    Otherwise:
!     If interpair distance is < CUTNB, group pair is included in list
!      as (+).
!     Else if CUTNB <distance< CTEXNB, do extended electrostatics
!      if requested (without inclusion in group-group list).
!     Else if distance >CTEXNB, ignore completely.
!
#if KEY_PARALLEL==1
      DO I = MYNODP,NACTG,NUMNOD
#else /**/
      DO I = 1,NACTG
#endif 
        AC1 = ACTVG(I)
! set the group1-dependent portions of the
! group table array exclusion pointers
        B1 = GREXPT(AC1)
        ZER = ORDCBN(AC1)
        MAXDIFG = NUEXCG(AC1)
        FIXED = IMOVEG(AC1) > 0
        MAC1X = RSCMX(AC1)
        MAC1Y = RSCMY(AC1)
        MAC1Z = RSCMZ(AC1)
        UU = HIMPRTN(ZER)
        SS = UU - NUMMP(ZER) + 1
        DO WHILE(SS <= UU)
          M = WRKAR(SS)
          QQ = 1
          AC2 = LSTM0(M)
          DO WHILE (QQ <= CNTN(M))
           IF(AC2 >= AC1) THEN
!  check to see if at least one group is moving
            IF((.NOT.FIXED).OR.(IMOVEG(AC2) <= 0)) THEN
!            Do exclusions
              DIFF = AC2 - AC1
              IF (DIFF <= MAXDIFG) THEN
                IF (DIFF == 0) THEN
                   EXCL14 = 0
                ELSE
                   ELEN = B1 + DIFF
                   EXCL14 = EXCLARG(ELEN)
                ENDIF
              ELSE
                EXCL14 = 1
              ENDIF
              IF(EXCL14 == 0) THEN
                NNNBG = NNNBG + 1
                JNBG(NNNBG) = -AC2
              ELSE  ! if not an excluded group pair
                XD = MAC1X - RSCMX(AC2)
                YD = MAC1Y - RSCMY(AC2)
                ZD = MAC1Z - RSCMZ(AC2)
                XX = XD*XD
                YY = YD*YD
                ZZ = ZD*ZD
                R2 = XX + YY + ZZ
                IF (R2 <= CTNBSQ) THEN
                  NNNBG = NNNBG + 1
                  JNBG(NNNBG) = AC2
!
!.......... Addition to group list ends here .........................
!   if moving and CTEXSQ > distance > CUTNB, do extended electro
                ELSE IF (LEXTND.AND.R2 < CTEXSQ) THEN ! (R2.LT.CTNBSQ)
!
!.......... Extended electrostatics starts here ......................
!          (this section shifted left two spaces)
!
!         If LEXTND and the two groups are further apart than
!         CUTNB but closer than CTEXSQ, then do extended
!         electrostatics INSTEAD of adding to the group pairlist.
!
               JS=IGPBS(AC2)+1
               JQ=IGPBS(AC2+1)
               XY=XD*YD
               YZ=YD*ZD
               ZX=ZD*XD
               R=SQRT(R2)
               R1=1.0/R
               R3=R1/R2
               R5=R3/R2
               R2X3=3.0/R2
               R2X5=5.0/R2
               R2X7=7.0/R2
!
               DO J = 1,2
!
                 IF (J == 1) THEN
                  IRST=AC1
                  JRST=AC2
                 ELSE
                  IRST=AC2
                  JRST=AC1
                  XD=-XD
                  YD=-YD
                  ZD=-ZD
                 END IF
                 DXT=RSDX(JRST)*R3
                 DYT=RSDY(JRST)*R3
                 DZT=RSDZ(JRST)*R3
                 DOT=(DXT*XD+DYT*YD+DZT*ZD)
!
                 IF (LQUAD) THEN
!
                  QXXT=RSQXX(JRST)*R5
                  QYYT=RSQYY(JRST)*R5
                  QZZT=RSQZZ(JRST)*R5
                  QXYT=RSQXY(JRST)*R5
                  QYZT=RSQYZ(JRST)*R5
                  QZXT=RSQZX(JRST)*R5
                  QXT=QXXT*XD+QXYT*YD+QZXT*ZD
!  changed YD to XD in line below --RJP
                  QYT=QYYT*YD+QYZT*ZD+QXYT*XD
                  QZT=QZZT*ZD+QZXT*XD+QYZT*YD
                  RQR=(QXT*XD+QYT*YD+QZT*ZD)/2.0
!
!
                  CGT=RSQ(JRST)*R1
!
                  RSPOT(IRST)=RSPOT(IRST)+CGT+DOT+RQR
!
                  CR3=CGT/R2
                  DOT=DOT*R2X3
                  RQR=RQR*R2X5
                  TEMP=CR3+DOT+RQR
!
                  RSFX(IRST)=RSFX(IRST)+DXT+QXT-TEMP*XD
                  RSFY(IRST)=RSFY(IRST)+DYT+QYT-TEMP*YD
                  RSFZ(IRST)=RSFZ(IRST)+DZT+QZT-TEMP*ZD
!
                  TEMP2=CR3*R2X3+DOT*R2X5+RQR*R2X7
                  DXT=DXT*R2X3+QXT*R2X5
                  DYT=DYT*R2X3+QYT*R2X5
                  DZT=DZT*R2X3+QZT*R2X5
!
                  RSGXX(IRST)=RSGXX(IRST)+TEMP2*XX-2.0*DXT*XD-TEMP+QXXT
                  RSGYY(IRST)=RSGYY(IRST)+TEMP2*YY-2.0*DYT*YD-TEMP+QYYT
                  RSGZZ(IRST)=RSGZZ(IRST)+TEMP2*ZZ-2.0*DZT*ZD-TEMP+QZZT
                  RSGXY(IRST)=RSGXY(IRST)+TEMP2*XY-DXT*YD-DYT*XD+QXYT
                  RSGYZ(IRST)=RSGYZ(IRST)+TEMP2*YZ-DYT*ZD-DZT*YD+QYZT
                  RSGZX(IRST)=RSGZX(IRST)+TEMP2*ZX-DZT*XD-DXT*ZD+QZXT

                 ELSE       ! (LQUAD)
!
                  CGT=RSQ(JRST)*R1
!
                  RSPOT(IRST)=RSPOT(IRST)+CGT+DOT
!
                  CR3=CGT/R2
                  DOT=DOT*R2X3
                  TEMP=CR3+DOT
!
                  RSFX(IRST)=RSFX(IRST)+DXT-TEMP*XD
                  RSFY(IRST)=RSFY(IRST)+DYT-TEMP*YD
                  RSFZ(IRST)=RSFZ(IRST)+DZT-TEMP*ZD
!
                  TEMP2=CR3*R2X3+DOT*R2X5
                  DXT=DXT*R2X3
                  DYT=DYT*R2X3
                  DZT=DZT*R2X3
!
                  RSGXX(IRST)=RSGXX(IRST)+TEMP2*XX-2.0*DXT*XD
                  RSGYY(IRST)=RSGYY(IRST)+TEMP2*YY-2.0*DYT*YD
                  RSGZZ(IRST)=RSGZZ(IRST)+TEMP2*ZZ-2.0*DZT*ZD
                  RSGXY(IRST)=RSGXY(IRST)+TEMP2*XY-DXT*YD-DYT*XD
                  RSGYZ(IRST)=RSGYZ(IRST)+TEMP2*YZ-DYT*ZD-DZT*YD
                  RSGZX(IRST)=RSGZX(IRST)+TEMP2*ZX-DZT*XD-DXT*ZD
!
                 END IF           ! (LQUAD)
!
               END DO              ! J = 1, 2
!
               NGPE=NGPE+1
!            (extended elect section shifted left 2 spaces)
!.......... Extended electrostatics section ends here ..............
!
!
                ELSE
                  NGPX=NGPX+1 ! unexcl pair beyond CTEXNB
                ENDIF                 ! (R2)
              ENDIF  ! if exclusion or not
            ELSE  ! if not moving
              NGPN = NGPN + 1    ! not moving
            ENDIF ! if moving or not
           ENDIF                  !AC2 > AC1
           AC2 = LSTN0(AC2)
           QQ = QQ + 1
          ENDDO         ! QQ (group in a surrounding cube)
        SS = SS +1
        ENDDO             ! SS (cubes in surroundings)
        INBLOG(AC1) = NNNBG
        IF ((NNNBG + NACTG) > MXJNBG) THEN
           NNNBG = MXJNBG + 1
           RETURN
        ENDIF
      ENDDO    ! over active groups (I)
! since non-bonded list has been filled only for active groups
! (rest of INBLOG(I) are zero) fill in the rest as appropriate
!
      DO I = 2,NGRP
       IF (INBLOG(I) == 0) INBLOG(I) = INBLOG(I-1)
      ENDDO
!
! ********************************************************************
! **************End search for group pairs****************************
! ********************************************************************
!
!---- Finish extended electrostatics ----------------------------------
!
!   The outer loop on IRST is not vectorizable, but the inner loop
!   on I is.
!
      IF (LEXTND) THEN
!
#if KEY_PARALLEL==1
         IF(NUMNOD > 1) THEN
            CALL GCOMB(RSPOT,NGRP)
            CALL GCOMB(RSFX ,NGRP)
            CALL GCOMB(RSFY ,NGRP)
            CALL GCOMB(RSFZ ,NGRP)
            CALL GCOMB(RSGXX,NGRP)
            CALL GCOMB(RSGYY,NGRP)
            CALL GCOMB(RSGZZ,NGRP)
            CALL GCOMB(RSGXY,NGRP)
            CALL GCOMB(RSGYZ,NGRP)
            CALL GCOMB(RSGZX,NGRP)
         ENDIF
#endif 
         DO KK = 1,NACTG
           IRST = ACTVG(KK)
            IS=IGPBS(IRST)+1
            IQ=IGPBS(IRST+1)
            DO I = IS,IQ
               XI=X(I)-RSCMX(IRST)
               YI=Y(I)-RSCMY(IRST)
               ZI=Z(I)-RSCMZ(IRST)
               QXT=RSGXX(IRST)*XI+RSGXY(IRST)*YI+RSGZX(IRST)*ZI
               QYT=RSGYY(IRST)*YI+RSGXY(IRST)*XI+RSGYZ(IRST)*ZI
               QZT=RSGZZ(IRST)*ZI+RSGZX(IRST)*XI+RSGYZ(IRST)*YI
               ATPOT(I)=ATPOT(I)+RSPOT(IRST)+((2.0*RSFX(IRST)+QXT)*XI+ &
                    (2.0*RSFY(IRST)+QYT)*YI+(2.0*RSFZ(IRST)+QZT)*ZI)/2.0
               ATFX(I)=ATFX(I)+RSFX(IRST)+QXT
               ATFY(I)=ATFY(I)+RSFY(IRST)+QYT
               ATFZ(I)=ATFZ(I)+RSFZ(IRST)+QZT
               IF (LGRAD) THEN
                  ATGXX(I)=ATGXX(I)+RSGXX(IRST)
                  ATGYY(I)=ATGYY(I)+RSGYY(IRST)
                  ATGZZ(I)=ATGZZ(I)+RSGZZ(IRST)
                  ATGXY(I)=ATGXY(I)+RSGXY(IRST)
                  ATGYZ(I)=ATGYZ(I)+RSGYZ(IRST)
                  ATGZX(I)=ATGZX(I)+RSGZX(IRST)
               END IF           ! (LGRAD)
            END DO              ! I = IS,IQ
         END DO                 ! IRST = 1, NGRP
!
!
!   This loop is vectorizable.
!
         DO KK = 1,NACTG
            IRST = ACTVG(KK)
            IS=IGPBS(IRST)+1
            IQ=IGPBS(IRST+1)
           DO I = IS,IQ
!C            I = ACTV(KK)
            CGT=CG(I)
            ATPOT(I)=ATPOT(I)*CGT/2.0
            ATFX(I)=ATFX(I)*CGT
            ATFY(I)=ATFY(I)*CGT
            ATFZ(I)=ATFZ(I)*CGT
            IF (LGRAD) THEN
               ATGXX(I)=ATGXX(I)*CGT
               ATGYY(I)=ATGYY(I)*CGT
               ATGZZ(I)=ATGZZ(I)*CGT
               ATGXY(I)=ATGXY(I)*CGT
               ATGYZ(I)=ATGYZ(I)*CGT
               ATGZX(I)=ATGZX(I)*CGT
            END IF              ! (LGRAD)
           END DO                 ! (I = 1, NATOM)
         END DO                 ! (I = 1, NATOM)
      END IF                    ! (LEXTND)
! ---------------------------------------------------------------------
!---- Flag success and print statistics --- ---------------------------
!
      CMPLTD=.TRUE.
!
      IF (PRNLEV > 4) THEN
!
         WRITE(OUTU,725) ' NBNDGG found: ',NNNBG, &
       ' GROUP PAIRS '
 725  FORMAT(A15,I9,A13)
         WRITE(OUTU,727) NGPN, &
        ' FIXED PAIRS NOT TESTED '
 727  FORMAT(I9,A)
!
            IF (LEXTND) THEN
              WRITE(OUTU,728) NGPE
              WRITE(OUTU,729) NGPX
 728  FORMAT(I9,' GROUP PAIRS USED EXTENDED ELECTRASTICS')
 729  FORMAT(I9,' GROUP PAIRS WERE BEYOND EXTE_D CUTOFFS')
            WRITE(OUTU,'(I7,A,I7,A,I7,A,I7,A,I7,A)') &
                 NGRP, ' groups:', &
                 IGPBD(1), '--no charges ', &
                 IGPBD(2), '--neutral ', &
                 IGPBD(3), '--charged ', &
                 IGPBD(4), '--ST2 '
!
            ENDIF
      ENDIF                    ! (PRNLEV >= 5)
      PRNLEV = PLVOLD
      if(QTIMING) then
       call seconds(etime,ctime)
       cumetime_cc = cumetime_cc + (etime-time1)
      endif
!
#endif /* (yesbycc2)*/
      RETURN
      end subroutine NBNDGG

      SUBROUTINE COMBINELIST(NLIST1,LIST1,NLIST2,LIST2, &
       NNNELE,NEWLIST)
!  this subroutine combines two sorted lists into one
! sorted list by taking the common elements of both
! (but removing duplicates)
!     
! passed
      IMPLICIT NONE
      INTEGER NLIST1,NLIST2,LIST1(*),LIST2(*),NEWLIST(*)
      INTEGER NNNELE
!  local 
      INTEGER SAVEELE,LASTSAVE,CC1,CC2,III
!
!       WRITE(6,*) 'NLIST1 ',NLIST1,' NLIST2 ',NLIST2
!      DO III = 1,NLIST1
!       WRITE(6,*) 'III ',III, ' LIST 1 ',LIST1(III)
!      ENDDO
!      DO III = 1,NLIST2
!       WRITE(6,*) 'III ',III, ' LIST 2 ',LIST2(III)
!      ENDDO

      CC1 = 1
      CC2 = 1
      LASTSAVE = 0
      NNNELE = 0
      DO WHILE((CC1 <= NLIST1).AND.(CC2.LE.NLIST2))
!       WRITE(6,*) 'CC1 ',CC1,' CC2 ',CC2
       IF(LIST1(CC1) < LIST2(CC2)) THEN
        SAVEELE = LIST1(CC1)
        CC1 = CC1 + 1
       ELSE IF(LIST1(CC1) > LIST2(CC2)) THEN
        SAVEELE = LIST2(CC2)
        CC2 = CC2 + 1
       ELSE !they are equal
        SAVEELE = LIST1(CC1)
        CC1 = CC1 + 1
        CC2 = CC2 + 1
       ENDIF
       IF(SAVEELE /= LASTSAVE) THEN
        NNNELE = NNNELE + 1
        NEWLIST(NNNELE) = SAVEELE
       ENDIF
       LASTSAVE = SAVEELE
      ENDDO
! check for stragglers
      IF(CC2 > NLIST2) THEN  !if this list is done, do the other
       DO III = CC1,NLIST1
        NNNELE = NNNELE + 1
        NEWLIST(NNNELE) = LIST1(III)
       ENDDO
      ELSE IF (CC1 > NLIST1) THEN !vice versa
       DO III = CC2,NLIST2
        NNNELE = NNNELE + 1
        NEWLIST(NNNELE) = LIST2(III)
       ENDDO
      ENDIF
      RETURN
      END
! 
      SUBROUTINE SPACSRNOPACK(X,Y,Z)
! 
  use chm_kinds
#if KEY_SPACDEC==1
  use dimens_fcm
  use parallel
  use spacdec
#endif 
      implicit none
      real(chm_real) X(*),Y(*),Z(*)
!
#if KEY_SPACDEC==1 /*spacdec10*/
      INTEGER ROUND,ATM,III
!
!      WRITE(6,*) 'IN SPACSR, NROUND IS ',NROUND
      DO ROUND = 1,NROUND
       YRATCNT = 0
!       WRITE(6,*) 'MYNODP ',MYNODP,' BEFORE: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
          CALL LOCINTCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,MYATCNT, &
       YRATCNT)
!        WRITE(6,*) 'MYNODP ',MYNODP,' AFTER LIC: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
!
       DO III = 1,MYATCNT
        ATM = MYATMAP(III)
!        WRITE(6,*) 'MYNODP ',MYNODP,' ROUND ',ROUND,' IND ',III,
!     & ' ATM ',ATM
        MYATMX(III) = X(ATM)
        MYATMY(III) = Y(ATM)
        MYATMZ(III) = Z(ATM)
       ENDDO 
       CALL LOCSPACCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,MYATCNT, &
       MYATMAP,MYATMX,MYATMY, &
       MYATMZ,YRATCNT,YRATMAP,YRATMX,YRATMY,YRATMZ)
       DO ATM = 1,YRATCNT
        X(YRATMAP(ATM)) = YRATMX(ATM)
        Y(YRATMAP(ATM)) = YRATMY(ATM)
        Z(YRATMAP(ATM)) = YRATMZ(ATM)
       ENDDO
!       WRITE(6,*) 'MYNODP ',MYNODP,' AFTER LSC: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
      ENDDO
!
!       IF(MYNOD == 3) THEN
!        DO ATM = 1,100
!          WRITE(6,*) 'AFTER LSC ATOM ',ATM,' X ',X(ATM),' Y ',Y(ATM),
!     & ' Z ',Z(ATM)
!        ENDDO
!       ENDIF   
!         DO ATM = 1,YRATCNT
!          WRITE(6,'(A13,I6,A5,I6,A6,I6,A4,F10.6,A4,F10.6,A4,F10.6)')
!     &  'RECEIVED: ME ',MYNODP,' IND ',ATM,' ATOM ',
!     & YRATMAP(ATM),' X ',YRATMX(ATM),' Y ',YRATMY(ATM),' Z ',
!     & YRATMZ(ATM)
!         ENDDO
!     
#endif /* (spacdec10)*/
      RETURN
      END
!
      SUBROUTINE FORCSRNOPACK(DX,DY,DZ)
!
  use chm_kinds
  use dimens_fcm
  use parallel
  use spacdec
      implicit none
! this routine sets up the exchange of forces
      INTEGER ROUND,ATM,III
      real(chm_real) DX(*),DY(*),DZ(*)
#if KEY_SPACDEC==1 /*spacdec10*/
!
!      WRITE(6,*) 'IN SPACSR, NROUND IS ',NROUND
      DO ROUND = 1,NROUND
       YRATCNT = 0
!       WRITE(6,*) 'MYNODP ',MYNODP,' BEFORE: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
          CALL LOCINTCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,MYATCNT, &
       YRATCNT)
!        WRITE(6,*) 'MYNODP ',MYNODP,' AFTER LIC: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
!
        CALL LOCMAPCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,MYATCNT, &
       MYATMAP,YRATCNT,YRATMAP)
       DO III = 1,YRATCNT
        ATM = YRATMAP(III)
!        WRITE(6,*) 'MYNODP ',MYNODP,' ROUND ',ROUND,' IND ',III,
!     & ' ATM ',ATM
        YOURDX(III) = DX(ATM)
        YOURDY(III) = DY(ATM)
        YOURDZ(III) = DZ(ATM)
       ENDDO 
       CALL LOCFORCCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,YRATCNT, &
       YOURDX,YOURDY,YOURDZ,MYATCNT,MINEDX,MINEDY,MINEDZ)
       DO ATM = 1,MYATCNT
        DX(MYATMAP(ATM)) = DX(MYATMAP(ATM)) + MINEDX(ATM)
        DY(MYATMAP(ATM)) = DY(MYATMAP(ATM)) + MINEDY(ATM)
        DZ(MYATMAP(ATM)) = DZ(MYATMAP(ATM)) + MINEDZ(ATM)
       ENDDO
!       WRITE(6,*) 'MYNODP ',MYNODP,' AFTER LSC: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
      ENDDO
!
!       IF(MYNOD == 3) THEN
!        DO ATM = 1,100
!          WRITE(6,*) 'AFTER LSC ATOM ',ATM,' X ',X(ATM),' Y ',Y(ATM),
!     & ' Z ',Z(ATM)
!        ENDDO
!       ENDIF
!         DO ATM = 1,YRATCNT
!          WRITE(6,'(A13,I6,A5,I6,A6,I6,A4,F10.6,A4,F10.6,A4,F10.6)')
!     &  'RECEIVED: ME ',MYNODP,' IND ',ATM,' ATOM ',
!     & YRATMAP(ATM),' X ',YRATMX(ATM),' Y ',YRATMY(ATM),' Z ',
!     & YRATMZ(ATM)
!         ENDDO
!
#endif /* (spacdec10)*/
      RETURN
      END
!
!     This routine works OK (could be fixed for zero index)
!     It is a standard blocking routine
!      SUBROUTINE FORCSR(DX,DY,DZ)
      SUBROUTINE FORCSRBLOCK(DX,DY,DZ)
!
!
  use chm_kinds
  use dimens_fcm
  use parallel
  use spacdec
      implicit none
! this routine sets up the exchange of forces
!
      real(chm_real) DX(*),DY(*),DZ(*)
#if KEY_SPACDEC==1 /*spacdec300*/
      INTEGER ROUND,ATM,III
      INTEGER HIPNT,LOPNT,COUNT,NODE2
      INTEGER END,BEGIN
!
!      WRITE(6,*) 'IN SPACSR, NROUND IS ',NROUND
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        HIPNT = YRMAPHI(NODE2)
        LOPNT = HIPNT - YRMAPMNY(NODE2) + 1
        YRATCNT = 0
        DO III = LOPNT,HIPNT
         ATM = YRMAPLST(III)
         YRATCNT = YRATCNT + 1
         YOURDX(YRATCNT) = DX(ATM)
         YOURDY(YRATCNT) = DY(ATM)
         YOURDZ(YRATCNT) = DZ(ATM)
        ENDDO
       MYATCNT = PRTATMMNY(NODE2)
!
       CALL LOCFORCCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,YRATCNT, &
       YOURDX,YOURDY,YOURDZ,MYATCNT,MINEDX,MINEDY,MINEDZ)
       END = PRTATMHI(NODE2)
       BEGIN = END - MYATCNT + 1
       COUNT = 0
       DO III = BEGIN,END
         ATM = PRTATMLST(III)
         COUNT = COUNT + 1
         DX(ATM) = DX(ATM) + MINEDX(COUNT)
         DY(ATM) = DY(ATM) + MINEDY(COUNT)
         DZ(ATM) = DZ(ATM) + MINEDZ(COUNT)
       ENDDO
!       WRITE(6,*) 'FORCSRPACK RECEVED ATOMS: ',MYATCNT,
!     & ' EXPECTED: ',PRTATMMNY(NODE2),' MYNODP ',MYNODP,' NODE2 ',NODE2
!       WRITE(6,*) 'MYNODP ',MYNODP,' AFTER LSC: ROUND ',ROUND,
!     & ' PART ',PARTNER(ROUND),' MYATCNT ',MYATCNT,' YRATCNT ',YRATCNT
      ENDDO
!
!       IF(MYNOD == 3) THEN
!        DO ATM = 1,100
!          WRITE(6,*) 'AFTER LSC ATOM ',ATM,' X ',X(ATM),' Y ',Y(ATM),
!     & ' Z ',Z(ATM)
!        ENDDO
!       ENDIF
!         DO ATM = 1,YRATCNT
!          WRITE(6,'(A13,I6,A5,I6,A6,I6,A4,F10.6,A4,F10.6,A4,F10.6)')
!     &  'RECEIVED: ME ',MYNODP,' IND ',ATM,' ATOM ',
!     & YRATMAP(ATM),' X ',YRATMX(ATM),' Y ',YRATMY(ATM),' Z ',
!     & YRATMZ(ATM)
!         ENDDO
!
#endif /* (spacdec300)*/
      RETURN
      END
!      
      SUBROUTINE SPACSRARG(ROUNDS,PARTNERS,XXX,YYY,ZZZ,SENDLEN, &
       SENDMAP,SENDX,SENDY,SENDZ,RECELEN,RECEMAP,RECEX, &
       RECEY,RECEZ)
!
  use chm_kinds
      implicit none
      INTEGER ROUNDS,PARTNERS(*),SENDLEN,SENDMAP(*),RECELEN, &
       RECEMAP(*)
      real(chm_real) SENDX(*),SENDY(*),SENDZ(*), &
           RECEX(*),RECEY(*),RECEZ(*)
      real(chm_real) XXX(*),YYY(*),ZZZ(*)
!
#if KEY_SPACDEC==1 /*spacdec10*/
! local variables
      INTEGER ROUND,III
      DO ROUND = 1,ROUNDS
        RECELEN = 0
        CALL LOCINTCOM(PARTNERS(ROUND)-1,PARTNERS(ROUND)-1,SENDLEN, &
       RECELEN)
        CALL LOCSPACCOM(PARTNERS(ROUND)-1,PARTNERS(ROUND)-1,SENDLEN, &
       SENDMAP,SENDX,SENDY,SENDZ,RECELEN,RECEMAP,RECEX,RECEY,RECEZ)
!        WRITE(6,*) 'IN SPACSRARG RECELEN IS ',RECELEN
        DO III = 1,RECELEN
         XXX(RECEMAP(III)) = RECEX(III)
         YYY(RECEMAP(III)) = RECEY(III)
         ZZZ(RECEMAP(III)) = RECEZ(III)
        ENDDO
      ENDDO  !loop over rounds
#endif /* (spacdec10)*/
      RETURN
      END
!CALL SPACSRPACK(NROUND,PARTNER,X,Y,Z,
!     & PRTATMLST,PRTATMHI,PRTATMMNY,NPARTATM,
!     & MYATMAP,MYATMX,MYATMY,MYATMZ,YRATMAP,YRATMX,
!     & YRATMY,YRATMZ)
!
      SUBROUTINE SPACSRPACKA(ROUNDS,PARTNERS,XXX,YYY,ZZZ, &
       PARTLIST,HIPNT,MNYPNT,LISTLEN,SENDMAP,SENDX,SENDY,SENDZ, &
       RECEMAP,RECEX,RECEY,RECEZ,NATOMX,PASSES)
!
  use chm_kinds
  use parallel
      implicit none
      INTEGER ROUNDS,PARTNERS(*),SENDMAP(*),RECEMAP(*)
      real(chm_real) SENDX(*),SENDY(*),SENDZ(*), &
           RECEX(*),RECEY(*),RECEZ(*)
      real(chm_real) XXX(*),YYY(*),ZZZ(*)
      INTEGER PARTLIST(*),HIPNT(*),MNYPNT(*),LISTLEN
      INTEGER NATOMX,PASSES !temporary
!
#if KEY_SPACDEC==1 /*spacdec10*/
! local variables
      INTEGER ROUND,III,SENDLEN,RECELEN,BEGIN,END,PARTICLE
      INTEGER NODE2,TOTSEND,TOTRECE
!
!      DO III = 1,NATOMX
!       WRITE(6,*) 'BEFORE COMM, PASS ',PASSES,' MYNODP ',MYNODP,
!     & ' ATOM ',III,' XXX ',XXX(III)
!      ENDDO
      TOTSEND = 0
      TOTRECE = 0
      DO ROUND = 1,ROUNDS
        NODE2 = PARTNERS(ROUND)
        END = HIPNT(NODE2)
        BEGIN = END - MNYPNT(NODE2) + 1
!        WRITE(6,*) 'PACKA ROUND ',ROUND,' NODE2 ',NODE2,' BEGIN ',
!     & BEGIN,' END ',END
        SENDLEN = 0
        DO III = BEGIN,END
         PARTICLE = PARTLIST(III) 
         TOTSEND = TOTSEND + 1
         SENDLEN = SENDLEN + 1
         SENDX(SENDLEN) = XXX(PARTICLE)
         SENDY(SENDLEN) = YYY(PARTICLE)
         SENDZ(SENDLEN) = ZZZ(PARTICLE)
         SENDMAP(SENDLEN) = PARTICLE
        ENDDO
        RECELEN = 0
        CALL LOCINTCOM(PARTNERS(ROUND)-1,PARTNERS(ROUND)-1,SENDLEN, &
       RECELEN)
!        WRITE(6,*) 'MYNODP ',MYNODP,' ROUND ',ROUND,' SENDLEN ',
!     & SENDLEN,' RECELEN ',RECELEN
        CALL LOCSPACCOM(PARTNERS(ROUND)-1,PARTNERS(ROUND)-1,SENDLEN, &
       SENDMAP,SENDX,SENDY,SENDZ,RECELEN,RECEMAP,RECEX,RECEY,RECEZ)
!        WRITE(6,*) 'IN SPACSRARG RECELEN IS ',RECELEN
        DO III = 1,RECELEN
         TOTRECE = TOTRECE + 1
!         WRITE(6,*) 'SPACSRPACKA ATM ',RECEMAP(III),' XXX ',
!     & RECEX(III)
         XXX(RECEMAP(III)) = RECEX(III)
         YYY(RECEMAP(III)) = RECEY(III)
         ZZZ(RECEMAP(III)) = RECEZ(III)
        ENDDO
      ENDDO  !loop over rounds
      WRITE(6,*) 'TOTAL ATOMS SENT IS ',TOTSEND,' MYNODP ',MYNODP
      WRITE(6,*) 'TOTAL ATOMS RECEIVED IS ',TOTRECE,' MYNODP ',MYNODP
#endif /* (spacdec10)*/
      RETURN
      END
!
!     old SPACSR() routine: too much blocking!
      SUBROUTINE SPACSRBLOCK(X,Y,Z)
!      SUBROUTINE SPACSR(X,Y,Z)
  use chm_kinds
#if KEY_SPACDEC==1
  use dimens_fcm
  use parallel
  use spacdec
#endif 
      implicit none
      real(chm_real) X(*),Y(*),Z(*)
#if KEY_SPACDEC==1 /*spacdec10*/
!
! local variables
      INTEGER ROUND,III,SENDLEN,RECELEN,BEGIN,END,PARTICLE
      INTEGER NODE2,LOPNT,HIPNT,ATOM,COUNT
!
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        END = PRTATMHI(NODE2)      !@ this is not good since node2 can be 0
        BEGIN = END - PRTATMMNY(NODE2) + 1
        SENDLEN = 0
        DO III = BEGIN,END
         PARTICLE = PRTATMLST(III)
         SENDLEN = SENDLEN + 1
         MYATMX(SENDLEN) = X(PARTICLE)
         MYATMY(SENDLEN) = Y(PARTICLE)
         MYATMZ(SENDLEN) = Z(PARTICLE)
        ENDDO
        RECELEN = YRMAPMNY(NODE2)
!        CALL LOCINTCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,SENDLEN,
!     & RECELEN)
        CALL LOCSPACCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,SENDLEN, &
       MYATMAP,MYATMX,MYATMY,MYATMZ,RECELEN,YRATMAP,YRATMX,YRATMY, &
       YRATMZ)
        HIPNT = YRMAPHI(NODE2)
        LOPNT = HIPNT - RECELEN + 1
        COUNT = 0
        DO III = LOPNT,HIPNT
         ATOM = YRMAPLST(III) 
         COUNT = COUNT + 1
         X(ATOM) = YRATMX(COUNT)
         Y(ATOM) = YRATMY(COUNT)
         Z(ATOM) = YRATMZ(COUNT)
        ENDDO
!      WRITE(6,*) 'SPACSRPACK RECEVED ATOMS: ',RECELEN,
!     & ' EXPECTED: ',YRMAPMNY(NODE2),' MYNODP ',MYNODP,' NODE2 ',NODE2 
      ENDDO  !loop over rounds
#endif /* (spacdec10)*/
      RETURN
      END
!
      SUBROUTINE SPACSRSET(X,Y,Z)
  use chm_kinds
#if KEY_SPACDEC==1
  use dimens_fcm
  use parallel
  use spacdec
#endif 
      implicit none
      real(chm_real) X(*),Y(*),Z(*)
#if KEY_SPACDEC==1 /*spacdec10*/
!
! local variables
      INTEGER ROUND,III,SENDLEN,RECELEN,BEGIN,END,PARTICLE
      INTEGER NODE2,NRECATM
      INTEGER HIPNT,LOPNT,INDEX,PNT  !temporary
!
      NRECATM = 0
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        END = PRTATMHI(NODE2)
        BEGIN = END - PRTATMMNY(NODE2) + 1
        SENDLEN = 0
        DO III = BEGIN,END
         PARTICLE = PRTATMLST(III)
         SENDLEN = SENDLEN + 1
         MYATMX(SENDLEN) = X(PARTICLE)
         MYATMY(SENDLEN) = Y(PARTICLE)
         MYATMZ(SENDLEN) = Z(PARTICLE)
         MYATMAP(SENDLEN) = PARTICLE
        ENDDO
        RECELEN = 0
        CALL LOCINTCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,SENDLEN, &
       RECELEN)
        CALL LOCSPACCOM(PARTNER(ROUND)-1,PARTNER(ROUND)-1,SENDLEN, &
       MYATMAP,MYATMX,MYATMY,MYATMZ,RECELEN,YRATMAP,YRATMX,YRATMY, &
       YRATMZ)
!        WRITE(6,*) 'IN SPACSRARG RECELEN IS ',RECELEN
        DO III = 1,RECELEN
         X(YRATMAP(III)) = YRATMX(III)
         Y(YRATMAP(III)) = YRATMY(III)
         Z(YRATMAP(III)) = YRATMZ(III)
! record the incoming map
         NRECATM = NRECATM + 1 
         YRMAPLST(NRECATM) = YRATMAP(III)
        ENDDO
        YRMAPHI(NODE2) = NRECATM
        YRMAPMNY(NODE2) = RECELEN
      ENDDO  !loop over rounds
!      WRITE(6,*) 'MYNODP ',MYNODP,' MUST RECEIVE ',NRECATM,' ATOMS '
      GOTO 1001
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        HIPNT = YRMAPHI(NODE2)
        LOPNT = HIPNT - YRMAPMNY(NODE2) + 1
        INDEX = 0
        DO PNT = LOPNT,HIPNT
         INDEX = INDEX + 1
!         WRITE(6,*) 'SENDING NODE ',NODE2,' INDEX ',INDEX,
!     & ' ATOM ', YRMAPLST(PNT)
        ENDDO
      ENDDO
 1001 CONTINUE
#endif /* (spacdec10)*/
      RETURN
      END
!
      SUBROUTINE PRCPUMAP(CMAP,NATOM,PASS,MYNODP)
!
  use chm_kinds
      implicit none
      INTEGER CMAP(*),III,NATOM,PASS,MYNODP
      DO III = 1,NATOM
       WRITE(6,*) 'ATOM ',III,' CPUMAP ',CMAP(III),' PASS ', &
       PASS,' MYNODP ',MYNODP
      ENDDO
      RETURN
      END
!
!      SUBROUTINE SPACSRNON(X,Y,Z)
      SUBROUTINE SPACSR_waitall_waitany(X,Y,Z)
!      SUBROUTINE SPACSR(X,Y,Z)
        use chm_kinds
        use dimens_fcm
        use parallel
        use spacdec
#if KEY_SPACEDEC==1
        use mpi      
#endif
        implicit none
      REAL*8 X(*),Y(*),Z(*)
#if KEY_SPACDEC==1 /*spacdec111*/

! local variables
      INTEGER ROUND,III,BEGIN,END,PARTICLE
      INTEGER LOPNT,HIPNT,ATOM,COUNT,ierr,i
      INTEGER :: totrec,totsen
      integer :: totrcnt,totscnt,node2
      integer*4 :: statigno(MPI_STATUS_SIZE)
      integer*4 :: SENDLEN,RECELEN,mpidp,from,to4,tag
      integer*4 :: mpicomw,ierr4,ihit,wstat(MPI_STATUS_SIZE)
      integer*4,allocatable,dimension(:) :: reqs,reqr
      real(chm_real),allocatable,dimension(:,:) :: xyzr,xyzs
      INTEGER NRCALL,NSCALL
!
!     new SPACSR: main communication routine in dynamics
!                 this routine is called on each step
!
!     these are structures that we can use here:
!     prtatmmny(cpu) = length of data to send to cpu
!     prtatmlst(i)   = mapping array for send data to get from X,Y,Z
!     xyzs(3,i)      = data from X,Y,Z according to mapping from prtatmlst
!     prtatmhi(cpu)  = last index in prtatmlst(i) that
!                      belongs to this cpu
!     yrmapmny(cpu)  = length of data to receive from cpu
!     yrmaplst(i)    = mapping array for received data to put into X,Y,Z
!     xyzr(3,i)      = data for X,Y,Z according to mapping from yrmaplst
!     yrmaphi(cpu)   = last index in yrmaplst(i) that
!                      belongs to this cpu
!
! first calculate the size to allocate for receives
! this should be done outside of this subroutine, 
! by spacsrset or something like that.
      
      allocate(reqr(NROUND),reqs(NROUND),stat=ierr)
      mpicomw=COMM_CHARMM
      mpidp=MPI_DOUBLE_PRECISION
      statigno(:)=MPI_STATUS_IGNORE(:)
      TOTREC = 0
      TOTSEN = 0
! calculate lengths first, so we can do the memory allocation
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        TOTREC = TOTREC + YRMAPMNY(NODE2)
        TOTSEN = TOTSEN + PRTATMMNY(NODE2)
      ENDDO
!
!      write(*,'(a,5i6)')'SPACSRfirst>me,totrec,totsen,nround=',mynod,totrec,totsen,nround
      allocate(xyzr(3,TOTREC),xyzs(3,TOTSEN),stat=ierr)
!
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
! calculate the send array
        END = PRTATMHI(NODE2)
        BEGIN = END - PRTATMMNY(NODE2) + 1
        DO III = BEGIN,END
         PARTICLE = PRTATMLST(III)
         xyzs(1,III) = x(PARTICLE)
         xyzs(2,III) = y(PARTICLE)
         xyzs(3,III) = z(PARTICLE)
        ENDDO
      ENDDO
!      write(*,'(a,5i6)')'SPACSRsecond>me,totrec,totsen,nround=',mynod,totrec,totsen,nround
!  post the receives
      NRCALL = 0
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        IF(NODE2 /= 0)then
        NRCALL = NRCALL + 1
        RECELEN = YRMAPMNY(NODE2)
        BEGIN = YRMAPHI(NODE2) - RECELEN + 1
        from=NODE2-1
!        write(50+mynod,'(10i6)')(yrmaplst(i),i=begin,yrmaphi(node2))
!        tag =NUMNOD*(NODE2-1) + MYNOD+1
        tag=1
!        write(50+mynod,'(a,6i6)')'irecv>me,node2,from,tag=',mynod,node2,from,tag
        call mpi_irecv(xyzr(1,BEGIN),RECELEN*3,mpidp,from,tag, &
             mpicomw,reqr(NRCALL),ierr4)        
        endif
      ENDDO
!      write(*,'(a,5i6)')'SPACSRthird>me,totrec,totsen,nround=',mynod,totrec,totsen,nround
!
!   post the sends
      NSCALL = 0
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        IF(NODE2 /= 0)then
        NSCALL = NSCALL + 1
        SENDLEN = PRTATMMNY(NODE2) 
        BEGIN = PRTATMHI(NODE2) - SENDLEN + 1
        to4=NODE2-1
!        write(50+mynod,'(a,6i6)')'SPACSRsend>me,node2,begin,sendlen=',mynod,node2,begin,sendlen
!        write(50+mynod,'(10i6)')(prtatmlst(i),i=begin,prtatmhi(node2))
!         tag = NUMNOD*(MYNOD) + NODE2
        tag=1
!        write(50+mynod,'(a,6i6)')'isend>me,node2,to4,tag=',mynod,node2,to4,tag
        call mpi_isend(xyzs(1,BEGIN),SENDLEN*3,mpidp,to4,tag, &
              mpicomw,reqs(NSCALL),ierr4)
        endif
      ENDDO
!      write(*,'(a,5i6)')'SPACSRfourth>me,totrec,totsen,nround=',mynod,totrec,totsen,nround

!      DO ROUND = 1,NROUND
!       write(6,*) 'mynode ',MYNOD+1,' ROUND ',ROUND,' REQR ',REQR(ROUND)
!      ENDDO
!      DO ROUND = 1,NROUND
!       write(6,*) 'mynode ',MYNOD+1,' ROUND ',ROUND,' REQS ',REQS(ROUND)
!      ENDDO
! now wait for the ones that finished receiving the data
!      write(50+mynod,'(a,6i6)')'beforeSPACSRwait>me,nrcall=',mynod,nrcall
!     the following MPI_STATUSES_IGNORE is OK to put directly in the call
!**!the waitall is for testing, we dont want it:
!**      call mpi_waitall(NRCALL,reqr,MPI_STATUSES_IGNORE,ierr4)
      DO ROUND = 1,NRCALL
!**      DO ROUND = 1,NROUND
!        write(6,*) 'ME ',MYNOD+1,' round ',round,' reqr '
         call mpi_waitany(NRCALL,reqr,ihit,statigno,ierr4)
!**         ihit=round
         NODE2 = PARTNER(ihit)
         if(node2 == 0)cycle
         HIPNT = YRMAPHI(NODE2)
         LOPNT = HIPNT - YRMAPMNY(NODE2) + 1
!         write(50+mynod,'(a,6i6)')'SPACSRwait>me,node2,lopnt,hipnt,ihit',mynod,node2,lopnt,hipnt,ihit
!        write(50+mynod,'(10i6)')(yrmaplst(i),i=lopnt,hipnt)
         DO III = LOPNT,HIPNT
            ATOM = YRMAPLST(III)
            X(ATOM) = xyzr(1,III)
            Y(ATOM) = xyzr(2,III)
            Z(ATOM) = xyzr(3,III)
         ENDDO
!        write(*,'(a,5i6)')'SPACSRfifth>me,totrec,totsen,nround=',
!     $     mynod,totrec,totsen,nround

!      WRITE(6,*) 'SPACSRPACK RECEVED ATOMS: ',RECELEN
!     & ' EXPECTED: ',YRMAPMNY(NODE2),' MYNODP ',MYNODP,' NODE2 ',NODE2
      ENDDO  !loop over rounds

!      I think we need to call this also for sends???? To clear up all the requests
      call mpi_waitall(NSCALL,reqs,MPI_STATUSES_IGNORE,ierr4)

      deallocate(reqr,reqs,xyzr,xyzs)
!      stop
#endif /* (spacdec111)*/
      return
      end
!
!     This one is for play!
      SUBROUTINE SPACSR_play(X,Y,Z)
        use chm_kinds
        use dimens_fcm
        use parallel
        use spacdec
#if KEY_SPACEDEC==1
        use mpi      
#endif
        implicit none
      REAL(chm_real) X(*),Y(*),Z(*)
#if KEY_SPACDEC==1 /*spacdec10*/
!
! local variables
      INTEGER ROUND,III,SENDLEN,RECELEN,PARTICLE
      INTEGER NODE2,LOPNT,HIPNT,ATOM,beginr,begins,ierr
!
      INTEGER :: totrec,totsen
      real(chm_real),allocatable,dimension(:,:) :: xyzr,xyzs
      INTEGER NRCALL,NSCALL
!
!     We need everything integer*4 here, so it works in
!     all environments ie. 32 bit and 64 bit
!
      INTEGER*4 M_STAT(MPI_STATUS_SIZE,2),IE,REQ(2),IX,TAG
      INTEGER*4 lto,lfrom,lls,llr,mpicomw,mpiint,mpidprec,i
!
      TAG=1
      mpiint=MPI_INTEGER
      mpidprec=MPI_DOUBLE_PRECISION
      mpicomw=COMM_CHARMM
!
!     here is the plan:
!OK       - start from original nbndcc+paral1
!OK       - use xyzs,xyzr with locspaccom
!       - import the blocking mpi calls from locspaccom:
!           - first in the same loop
!           - separate the loops
!       - replace the blocking with nonblockin mpi
!
      TOTREC = 0
      TOTSEN = 0
! calculate lengths first, so we can do the memory allocation
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        TOTREC = TOTREC + YRMAPMNY(NODE2)
        TOTSEN = TOTSEN + PRTATMMNY(NODE2)
      ENDDO
      allocate(xyzr(TOTREC,3),xyzs(TOTSEN,3),stat=ierr)
!
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
! calculate the send array
        hipnt = PRTATMHI(NODE2)
        lopnt = hipnt - PRTATMMNY(NODE2) + 1
        DO III = lopnt,hipnt
         PARTICLE = PRTATMLST(III)
         xyzs(III,1) = x(PARTICLE)
         xyzs(III,2) = y(PARTICLE)
         xyzs(III,3) = z(PARTICLE)
        ENDDO
      ENDDO
!
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        SENDLEN = PRTATMMNY(NODE2)
        begins = prtatmhi(node2) - sendlen + 1
        RECELEN = YRMAPMNY(NODE2)
        beginr = yrmaphi(node2) - recelen + 1
        lfrom=node2-1
        lto=node2-1
        lls=sendlen
        llr=recelen
!        CALL LOCSPACCOM(to,from,SENDLEN,
!     $     MYATMAP,xyzs(begins,1),xyzs(begins,2),xyzs(begins,3),RECELEN,
!     $     YRATMAP,xyzr(beginr,1),xyzr(beginr,2),xyzr(beginr,3),.false.)
!
        IX=0
        IF(LTO > -1)THEN
           ix=ix+1
           CALL MPI_ISEND(xyzs(begins,1),lls,mpidprec,lto,tag,mpicomw,req(ix),ie)
        ENDIF
        IF(LFROM > -1)THEN
           IX=IX+1
           CALL MPI_IRECV(xyzr(beginr,1),llr,mpidprec,lfrom,tag,mpicomw,req(ix),ie)
        ENDIF
!
        IF((lto > -1).or.(lfrom.gt.-1))then
           CALL MPI_WAITALL(ix,req,m_stat,ie)
        ENDIF
!
        IX=0
        IF(LTO > -1)THEN
           ix=ix+1
           CALL MPI_ISEND(xyzs(begins,2),lls,mpidprec,lto,tag,mpicomw,req(ix),ie)
        ENDIF
        IF(LFROM > -1)THEN
           IX=IX+1
           CALL MPI_IRECV(xyzr(beginr,2),llr,mpidprec,lfrom,tag,mpicomw,req(ix),ie)
        ENDIF
!
        IF((lto > -1).or.(lfrom.gt.-1))then
           CALL MPI_WAITALL(ix,req,m_stat,ie)
        ENDIF
!
        IX=0
        IF(LTO > -1)THEN
           ix=ix+1
           CALL MPI_ISEND(xyzs(begins,3),lls,mpidprec,lto,tag,mpicomw,req(ix),ie)
        ENDIF
        IF(LFROM > -1)THEN
           IX=IX+1
           CALL MPI_IRECV(xyzr(beginr,3),llr,mpidprec,lfrom,tag,mpicomw,req(ix),ie)
        ENDIF
!
        IF((lto > -1).or.(lfrom.gt.-1))then
           CALL MPI_WAITALL(ix,req,m_stat,ie)
        ENDIF
!
      ENDDO
!
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        RECELEN = YRMAPMNY(NODE2)
        hipnt = yrmaphi(node2)
        lopnt = hipnt - recelen + 1
        DO III = lopnt,hipnt
         ATOM = YRMAPLST(III) 
         X(ATOM) = xyzr(iii,1)
         Y(ATOM) = xyzr(iii,2)
         Z(ATOM) = xyzr(iii,3)
        ENDDO
      ENDDO
!
      deallocate(xyzr,xyzs,stat=ierr)
#endif /* (spacdec10)*/
      RETURN
      END
!
      SUBROUTINE SPACSR_trouble(X,Y,Z)
!
!     here is the plan:
!       - start from original nbndcc+paral1
!       - use xyzs,xyzr with locspaccom
!       - import the blocking mpi calls from locspaccom
!       - replace the blocking with nonblockin mpi
!
!     here is another plan:
!       - start from original nbndcc+paral1
!       - import mpi stuff from locspaccom into this routine
!       - introduce xyzs,xyzr
!       - replace blocking mpi with nonblocking
!
        use chm_kinds
        use dimens_fcm
        use parallel
        use spacdec
#if KEY_SPACDEC==1
        use mpi       
#endif
        implicit none
      REAL(chm_real) X(*),Y(*),Z(*)
#if KEY_SPACDEC==1 /*spacdec111*/

! local variables
      INTEGER ROUND,III,BEGIN,END,PARTICLE
      INTEGER LOPNT,HIPNT,ATOM,COUNT,ierr,i
      INTEGER :: totrec,totsen
      integer :: totrcnt,totscnt,node2
      integer*4 :: statigno(MPI_STATUS_SIZE)
      integer*4 :: SENDLEN,RECELEN,mpidp,from,to4,tag
      integer*4 :: mpicomw,ierr4,ihit,wstat(MPI_STATUS_SIZE)
      integer*4,allocatable,dimension(:) :: reqs,reqr
      real(chm_real),allocatable,dimension(:,:) :: xyzr,xyzs
      INTEGER NRCALL,NSCALL
!
!     new SPACSR: main communication routine in dynamics
!                 this routine is called on each step
!
!     these are structures that we can use here:
!     prtatmmny(cpu) = length of data to send to cpu
!     prtatmlst(i)   = mapping array for send data to get from X,Y,Z
!     xyzs(3,i)      = data from X,Y,Z according to mapping from prtatmlst
!     prtatmhi(cpu)  = last index in prtatmlst(i) that
!                      belongs to this cpu
!     yrmapmny(cpu)  = length of data to receive from cpu
!     yrmaplst(i)    = mapping array for received data to put into X,Y,Z
!     xyzr(3,i)      = data for X,Y,Z according to mapping from yrmaplst
!     yrmaphi(cpu)   = last index in yrmaplst(i) that
!                      belongs to this cpu
!
! first calculate the size to allocate for receives
! this should be done outside of this subroutine, 
! by spacsrset or something like that.
      
      allocate(reqr(NROUND),reqs(NROUND),stat=ierr)
      mpicomw=COMM_CHARMM
      mpidp=MPI_DOUBLE_PRECISION
      statigno(:)=MPI_STATUS_IGNORE(:)
      TOTREC = 0
      TOTSEN = 0
! calculate lengths first, so we can do the memory allocation
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        TOTREC = TOTREC + YRMAPMNY(NODE2)
        TOTSEN = TOTSEN + PRTATMMNY(NODE2)
      ENDDO
!
      write(*,'(a,5i6)')'SPACSRfirst>me,totrec,totsen,nround=', &
           mynod,totrec,totsen,nround
      allocate(xyzr(3,TOTREC),xyzs(3,TOTSEN),stat=ierr)
!
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
! calculate the send array
        END = PRTATMHI(NODE2)
        BEGIN = END - PRTATMMNY(NODE2) + 1
        DO III = BEGIN,END
         PARTICLE = PRTATMLST(III)
         xyzs(1,III) = x(PARTICLE)
         xyzs(2,III) = y(PARTICLE)
         xyzs(3,III) = z(PARTICLE)
        ENDDO
      ENDDO
      write(*,'(a,5i6)')'SPACSRsecond>me,totrec,totsen,nround=', &
           mynod,totrec,totsen,nround
!  post the receives
      NRCALL = 0
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        IF(NODE2 /= 0)then
        NRCALL = NRCALL + 1
        RECELEN = YRMAPMNY(NODE2)
        BEGIN = YRMAPHI(NODE2) - RECELEN + 1
        from=NODE2-1
!        write(50+mynod,'(10i6)')(yrmaplst(i),i=begin,yrmaphi(node2))
        tag =NUMNOD*(NODE2-1) + MYNOD+1
        write(50+mynod,'(a,6i6)')'irecv>me,node2,from,tag=', &
             mynod,node2,from,tag
        call mpi_irecv(xyzr(1,BEGIN),RECELEN*3,mpidp,from,tag, &
             mpicomw,reqr(NRCALL),ierr4)        
        endif
      ENDDO
      write(*,'(a,5i6)')'SPACSRthird>me,totrec,totsen,nround=', &
           mynod,totrec,totsen,nround
!
!   post the sends
      NSCALL = 0
      DO ROUND = 1,NROUND
        NODE2 = PARTNER(ROUND)
        IF(NODE2 /= 0)then
        NSCALL = NSCALL + 1
        SENDLEN = PRTATMMNY(NODE2) 
        BEGIN = PRTATMHI(NODE2) - SENDLEN + 1
        to4=NODE2-1
        write(50+mynod,'(a,6i6)')'SPACSRsend>me,node2,begin,sendlen=', &
             mynod,node2,begin,sendlen
!        write(50+mynod,'(10i6)')(prtatmlst(i),i=begin,prtatmhi(node2))
         tag = NUMNOD*(MYNOD) + NODE2
         write(50+mynod,'(a,6i6)')'isend>me,node2,to4,tag=', &
              mynod,node2,to4,tag
         call mpi_isend(xyzs(1,BEGIN),SENDLEN*3,mpidp,to4,tag, &
              mpicomw,reqs(NSCALL),ierr4)
        endif
      ENDDO
      write(*,'(a,5i6)')'SPACSRfourth>me,totrec,totsen,nround=', &
           mynod,totrec,totsen,nround

      DO ROUND = 1,NROUND
       write(6,*) 'mynode ',MYNOD+1,' ROUND ',ROUND,' REQR ',REQR(ROUND)
      ENDDO
      DO ROUND = 1,NROUND
       write(6,*) 'mynode ',MYNOD+1,' ROUND ',ROUND,' REQS ',REQS(ROUND)
      ENDDO
! now wait for the ones that finished receiving the data
      write(50+mynod,'(a,6i6)')'beforeSPACSRwait>me,nrcall=',mynod,nrcall
!     the following MPI_STATUSES_IGNORE is OK to put directly in the call
      call mpi_waitall(NRCALL,reqr,MPI_STATUSES_IGNORE,ierr4)
!      DO ROUND = 1,NRCALL
      DO ROUND = 1,NROUND
!        write(6,*) 'ME ',MYNOD+1,' round ',round,' reqr '
!        call mpi_waitany(NRCALL,reqr,ihit,statigno,ierr4)
        ihit=round
        NODE2 = PARTNER(ihit)
        if(node2 == 0)cycle
        HIPNT = YRMAPHI(NODE2)
        LOPNT = HIPNT - YRMAPMNY(NODE2) + 1
        write(50+mynod,'(a,6i6)')'SPACSRwait>me,node2,lopnt,hipnt,ihit', &
             mynod,node2,lopnt,hipnt,ihit
!        write(50+mynod,'(10i6)')(yrmaplst(i),i=lopnt,hipnt)
        DO III = LOPNT,HIPNT
         ATOM = YRMAPLST(III)
         X(ATOM) = xyzr(1,III)
         Y(ATOM) = xyzr(2,III)
         Z(ATOM) = xyzr(3,III)
        ENDDO
!        write(*,'(a,5i6)')'SPACSRfifth>me,totrec,totsen,nround=',
!     $     mynod,totrec,totsen,nround

!      WRITE(6,*) 'SPACSRPACK RECEVED ATOMS: ',RECELEN
!     & ' EXPECTED: ',YRMAPMNY(NODE2),' MYNODP ',MYNODP,' NODE2 ',NODE2
      ENDDO  !loop over rounds
      deallocate(reqr,reqs,xyzr,xyzs)
!      stop
#endif /* (spacdec111)*/
      RETURN
      END
!
!     new SPACSR() routine: from the old one, but waitany()
      SUBROUTINE SPACSR(X,Y,Z)
!      SUBROUTINE SPACSR_oldbutwaitany(X,Y,Z)
        use chm_kinds
        use dimens_fcm
        use parallel
        use spacdec
#if KEY_SPACDEC==1
        use mpi       
#endif

        implicit none
        real(chm_real) X(*),Y(*),Z(*)
#if KEY_SPACDEC==1 /*spacdec10wa*/

        ! local variables
        integer,parameter :: maxirounds = 2000*6 !(6=3rec+3sen)
        INTEGER ROUND,III,SENDLEN,RECELEN,BEGIN,IEND,PARTICLE
        INTEGER NODE2,LOPNT,HIPNT,ATOM,COUNT
        integer*4 tag,sizer,to,from,sizes,ie,mpd,mcw,ixr,ixs
        integer*4 ms(mpi_status_size,maxirounds),ihit
        integer*4,allocatable,dimension(:) :: rqs, rqr
        integer*4 :: statigno(MPI_STATUS_SIZE)
        integer,allocatable,dimension(:) :: rqs2cpu,rqr2cpu
        type arofar
           real(chm_real),allocatable,dimension(:) :: b
        end type arofar
        type(arofar),allocatable,dimension(:) :: sxyz,rxyz
        !
        tag=1
        mcw=mpi_comm_world
        mpd=mpi_double_precision
        statigno(:)=MPI_STATUS_IGNORE(:)

        !  We rxyz and sxyz to avoid overwrites!
        allocate(sxyz(numnod),rxyz(numnod))
        allocate(rqs(numnod),rqr(numnod))
        allocate(rqs2cpu(numnod),rqr2cpu(numnod))

        ixr=0
        ixs=0
        DO ROUND = 1,NROUND
           NODE2 = PARTNER(ROUND)
           IF (NODE2==0) CYCLE       ! we cannot address 0 - NEVER
           IEND = PRTATMHI(NODE2)    ! /but it was forgiving - fix everywhere!!!/
           SENDLEN = PRTATMMNY(NODE2)
           RECELEN = YRMAPMNY(NODE2)
           BEGIN = IEND - SENDLEN + 1

           !write(*,*)'SPACSR-allocate>me,round,node2=',mynod,round,node2
           allocate(sxyz(node2)%b(3*sendlen),rxyz(node2)%b(3*recelen))

           COUNT=0
           DO III = BEGIN,IEND
              PARTICLE = PRTATMLST(III)
              COUNT=COUNT+1
              sxyz(node2)%b(COUNT) = X(PARTICLE)
              COUNT=COUNT+1
              sxyz(node2)%b(COUNT) = Y(PARTICLE)
              COUNT=COUNT+1
              sxyz(node2)%b(COUNT) = Z(PARTICLE)
           ENDDO

           ! I*4 copy:
           to=node2-1
           from=node2-1 ! FROM in this loop is the same as TO
           sizes=sendlen*3
           sizer=recelen*3

           if(to > -1)then  ! this happens ALWAYS  ???!!!
              ixs=ixs+1
              rqs2cpu(ixs)=to
              call mpi_isend(sxyz(node2)%b,sizes,mpd,to,tag,mcw,rqs(ixs),ie)
           endif
           if(from > -1)then
              ixr=ixr+1
              rqr2cpu(ixr)=from
              call mpi_irecv(rxyz(node2)%b,sizer,mpd,from,tag,mcw,rqr(ixr),ie)
           endif

        ENDDO  !loop over rounds

        ! for waitany: do we need to separate
        !              receive and send requests !!!????
        
        ! the following mpi_waitall() maybe needs to be called
        ! with the same size of the request arrays????
        !call mpi_waitall(ix,rq,ms,ie) ! this is also not so bad ???
        !write(*,*)'SPACSR-waitall>me,ix,ie=',mynod,ix,ie

        DO ROUND=1,ixr
           call mpi_waitany(ixr,rqr,ihit,statigno,ie)
           !write(*,*)'SPACSR-waitany>me,ixr,ihit=',mynod,ixr,ihit
           NODE2 = rqr2cpu(ihit)+1     ! ihit starts with 0 or 1 ???
           RECELEN = YRMAPMNY(NODE2)
           HIPNT = YRMAPHI(NODE2)
           LOPNT = HIPNT - RECELEN + 1
           COUNT = 0
           DO III = LOPNT,HIPNT
              ATOM = YRMAPLST(III)
              COUNT = COUNT + 1
              X(ATOM) = rxyz(node2)%b(COUNT)
              COUNT = COUNT + 1
              Y(ATOM) = rxyz(node2)%b(COUNT)
              COUNT = COUNT + 1
              Z(ATOM) = rxyz(node2)%b(COUNT)
           ENDDO
        ENDDO

        ! we should empty send requests - HOW?
        ! call waitany.... in the loop ??? can we do a shortcut??
        ! maybe this should be above the receives???

        do round=1,ixs
           call mpi_waitany(ixs,rqs,ihit,statigno,ie)
        enddo

        deallocate(sxyz,rxyz)
        deallocate(rqs,rqr)
        deallocate(rqs2cpu,rqr2cpu)

#endif /* (spacdec10wa)*/
        RETURN

      END SUBROUTINE SPACSR
!
!     nonblocking force communications with mpi_waitany()
      SUBROUTINE FORCSR(DX,DY,DZ)
!
        use chm_kinds
        use dimens_fcm
        use parallel
        use spacdec
#if KEY_SPACDEC==1
        use mpi       
#endif

        implicit none
! this routine sets up the exchange of forces
!
        real(chm_real) DX(*),DY(*),DZ(*)
#if KEY_SPACDEC==1 /*spacdec3001*/
        integer,parameter :: maxirounds = 2000*2 !(2=1rec+1sen)
        INTEGER ROUND,ATM,III
        INTEGER HIPNT,LOPNT,COUNT,NODE2
        INTEGER IEND,BEGIN
!
        integer*4 tag,sizer,to,from,sizes,ie,mpd,mcw,ixr,ixs
        integer*4 ms(mpi_status_size,maxirounds),ihit
        integer*4,allocatable,dimension(:) :: rqs, rqr
        integer*4 :: statigno(MPI_STATUS_SIZE)
        integer,allocatable,dimension(:) :: rqs2cpu,rqr2cpu
        type arofar
           real(chm_real),allocatable,dimension(:) :: b
        end type arofar
        type(arofar),allocatable,dimension(:) :: sxyz,rxyz
        !
        tag=1
        mcw=mpi_comm_world
        mpd=mpi_double_precision
        statigno(:)=MPI_STATUS_IGNORE(:)
        !Use rxyz and sxyz to avoid overwrites during communication!
        allocate(sxyz(numnod),rxyz(numnod))
        allocate(rqs(numnod),rqr(numnod))
        allocate(rqs2cpu(numnod),rqr2cpu(numnod))
        ixr=0
        ixs=0
        DO ROUND = 1,NROUND
           NODE2 = PARTNER(ROUND)
           IF (NODE2==0) CYCLE
           HIPNT = YRMAPHI(NODE2)
           LOPNT = HIPNT - YRMAPMNY(NODE2) + 1
           YRATCNT = YRMAPMNY(NODE2)
           MYATCNT = PRTATMMNY(NODE2)
           allocate(sxyz(node2)%b(3*yratcnt),rxyz(node2)%b(3*myatcnt))
           COUNT=0
           DO III = LOPNT,HIPNT
              ATM = YRMAPLST(III)
              COUNT = COUNT + 1
              sxyz(node2)%b(COUNT) = DX(ATM)
              COUNT=COUNT+1
              sxyz(node2)%b(COUNT) = DY(ATM)
              COUNT=COUNT+1          
              sxyz(node2)%b(COUNT) = DZ(ATM)
           ENDDO
           !
           ! I*4 copy:
           to=node2-1
           from=node2-1 ! FROM in this loop is the same as TO
           sizes=yratcnt*3
           sizer=myatcnt*3

           if(to > -1)then  ! this happens ALWAYS  ???!!!
              ixs=ixs+1
              rqs2cpu(ixs)=to
              call mpi_isend(sxyz(node2)%b,sizes,mpd,to,tag,mcw,rqs(ixs),ie)
           endif
           if(from > -1)then
              ixr=ixr+1
              rqr2cpu(ixr)=from
              call mpi_irecv(rxyz(node2)%b,sizer,mpd,from,tag,mcw,rqr(ixr),ie)
           endif
        ENDDO

        do round=1,ixr
           call mpi_waitany(ixr,rqr,ihit,statigno,ie)
           !write(*,*)'FORCSR-waitany>me,ixr,ihit=',mynod,ixr,ihit
           NODE2 = rqr2cpu(ihit)+1
           IEND= PRTATMHI(NODE2)
           BEGIN = IEND - PRTATMMNY(NODE2) + 1
           COUNT = 0
           DO III = BEGIN,IEND
              ATM = PRTATMLST(III)
              COUNT = COUNT + 1
              DX(ATM) = DX(ATM) + rxyz(node2)%b(COUNT)
              COUNT = COUNT + 1
              DY(ATM) = DY(ATM) + rxyz(node2)%b(COUNT)
              COUNT = COUNT + 1
              DZ(ATM) = DZ(ATM) + rxyz(node2)%b(COUNT)
           ENDDO
        ENDDO

        ! we should empty send requests - HOW?
        ! call waitany.... in the loop ??? can we do a shortcut?
        ! maybe this should be above the receives???

        do round=1,ixs
           call mpi_waitany(ixs,rqs,ihit,statigno,ie)
        enddo

        deallocate(sxyz,rxyz)
        deallocate(rqs,rqr)
        deallocate(rqs2cpu,rqr2cpu)
!
#endif /* (spacdec3001)*/
        RETURN
      END SUBROUTINE FORCSR

